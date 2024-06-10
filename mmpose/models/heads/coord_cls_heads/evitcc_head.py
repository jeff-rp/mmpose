# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn
import torch.nn.functional as F

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]

class ResLiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads_ratio: float = 1.0,
        dim=8,
        act_func=(None, None),
        scales=(5,),
        eps=1.0e-15,
    ):
        super(ResLiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        act_func = (act_func, act_func)

        self.dim = dim
        self.qkv = nn.Conv2d(in_channels, 3 * total_dim, 1, bias=False)
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, scale, padding=scale // 2,
                          groups=3 * total_dim, bias=False),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=False),
            )
            for scale in (5,)
        ])
        self.kernel_func = nn.ReLU(inplace=False)

        self.proj = nn.Sequential(
            nn.Conv2d(total_dim * (1 + len(scales)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return x + out

class ResMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6):
        super(ResMBConv, self).__init__()

        mid_channels = round(in_channels * expand_ratio)
        self.inverted_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                      padding=kernel_size//2, groups=mid_channels, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.point_conv(self.depth_conv(self.inverted_conv(x)))

class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales=(5,)
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResLiteMLA(
            in_channels=in_channels,
            out_channels=in_channels,
            heads_ratio=heads_ratio,
            dim=dim,
            scales=scales
        )
        self.local_module = ResMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x

@MODELS.register_module()
class EVitCCHead(BaseHead):
    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)
        
        self.evit_block = EfficientViTBlock(out_channels, dim=16, expand_ratio=4)

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, 256, bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.cls_x = nn.Linear(256, W, bias=False)
        self.cls_y = nn.Linear(256, H, bias=False)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        feats = feats[-1]

        feats = self.final_layer(feats)  # -> B, K, H, W
        feats = self.evit_block(feats)

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)  # -> B, K, hidden

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {}
    ) -> InstanceList:
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):

                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg