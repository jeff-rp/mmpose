from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates, flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                 Predictions)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]

@MODELS.register_module()
class DSNTRLEHead(BaseHead):
    def __init__(self,
                 in_channels: int,
                 heatmap_size: Tuple[int, int],
                 num_joints: int,
                 loss: ConfigType = dict(
                    type='MultipleLossWrapper',
                    losses=[dict(type='RLELoss', use_target_weight=True),
                            dict(type='JSDiscretLoss', use_target_weight=True)]
                 ),
                 dist_w: float = 1.0,
                 use_depthwise_deconv = True,
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.num_joints = num_joints
        self.loss_module = MODELS.build(loss)
        self.dist_w = dist_w
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if use_depthwise_deconv:
            self.deconv = nn.Sequential(
                build_upsample_layer(dict(
                    type='deconv',
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                    groups=in_channels)
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                build_conv_layer(dict(
                    type='Conv2d',
                    in_channels=in_channels,
                    out_channels=256,
                    kernel_size=1)
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        else:
            self.deconv = nn.Sequential(
                build_upsample_layer(dict(
                    type='deconv',
                    in_channels=in_channels,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False)
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.final_layer = build_conv_layer(dict(
            type='Conv2d',
            in_channels=256,
            out_channels=num_joints,
            kernel_size=1)
        )
        W, H = heatmap_size
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, 1, W) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H, 1) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

        self.conv_b = build_conv_layer(dict(
            type='Conv2d',
            in_channels=in_channels,
            out_channels=256,
            kernel_size=1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_joints * 2)

    def _linear_expectation(self, heatmaps: Tensor,
                            linspace: Tensor) -> Tensor:
        B, N, _, _ = heatmaps.shape
        heatmaps = heatmaps.mul(linspace).reshape(B, N, -1)
        expectation = torch.sum(heatmaps, dim=2, keepdim=True)

        return expectation

    def _flat_softmax(self, featmaps: Tensor) -> Tensor:
        _, N, H, W = featmaps.shape

        featmaps = featmaps.reshape(-1, N, H * W)
        heatmaps = F.softmax(featmaps, dim=2)

        return heatmaps.reshape(-1, N, H, W)

    def forward(self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        x = feats[-1]
        x1 = self.deconv(x)
        x1 = self.final_layer(x1)
        heatmaps = self._flat_softmax(x1)

        pred_x = self._linear_expectation(heatmaps, self.linspace_x)
        pred_y = self._linear_expectation(heatmaps, self.linspace_y)

        x2 = self.conv_b(x)
        x2 = self.gap(x2)
        sigma = self.fc(torch.flatten(x2, 1))
        sigma = sigma.reshape(-1, self.num_joints, 2)
        coords = torch.cat([pred_x, pred_y, sigma], dim=-1)

        return coords, heatmaps
 
    def predict(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats
            _batch_coords, _batch_heatmaps, = self.forward(_feats)
            _batch_coords_flip, _batch_heatmaps_flip = self.forward(_feats_flip)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size
            )
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode='heatmap',
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False)
            )
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_coords, batch_heatmaps = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        pred_coords, pred_heatmaps = self.forward(inputs)
        keypoint_labels = torch.cat([d.gt_instance_labels.keypoint_labels
                                     for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights
                                      for d in batch_data_samples])
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])

        pred_sigma = pred_coords[..., 2:4]
        pred_coords = pred_coords[..., :2]
  
        losses = dict()

        loss_k = self.loss_module.loss_modules[0](pred_coords, pred_sigma, keypoint_labels,
                                                  keypoint_weights.unsqueeze(-1))
        loss_r = self.loss_module.loss_modules[1](pred_heatmaps, gt_heatmaps, keypoint_weights)
        loss_r *= self.dist_w

        losses.update(loss_k=loss_k)
        losses.update(loss_r=loss_r)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32)
        )
        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
            dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg