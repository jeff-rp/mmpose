import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import ConfigType, OptConfigType, Features, OptSampleList, Predictions
from mmpose.utils.tensor_utils import to_numpy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.evaluation.functional import pose_pck_accuracy
from ..base_head import BaseHead
from typing import Sequence, Tuple

class UpSampleLayer(nn.Module):
    def __init__(self, mode="bilinear", factor=2, align_corners=False):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1: return x
        return F.interpolate(x, size=None, scale_factor=self.factor, mode=self.mode,
                             align_corners=self.align_corners)

class ResMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6):
        super(ResMBConv, self).__init__()

        mid_channels = round(in_channels * expand_ratio)
        self.inverted_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                      padding=kernel_size//2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.point_conv(self.depth_conv(self.inverted_conv(x)))

class SegHead(nn.Module):
    def __init__(self,
                 in_channel_list: list[int],
                 head_width: int,
                 head_depth: int,
                 expand_ratio: float,
                 final_expand: float,
                 n_classes: int):
        super(SegHead, self).__init__()

        self.inputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel_list[0], head_width, 1, bias=False),
                nn.BatchNorm2d(head_width)
            ),
            nn.Sequential(
                nn.Conv2d(in_channel_list[1], head_width, 1, bias=False),
                nn.BatchNorm2d(head_width),
                UpSampleLayer(factor=2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channel_list[2], head_width, 1, bias=False),
                nn.BatchNorm2d(head_width),
                UpSampleLayer(factor=4)
            ),
        ])

        middle = []
        for _ in range(head_depth):
            middle.append(ResMBConv(head_width, head_width, expand_ratio=expand_ratio))
        self.middle = nn.Sequential(*middle)

        expand_channels = round(head_width * final_expand)
        self.output = nn.Sequential(
            nn.Conv2d(head_width, expand_channels, 1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(round(head_width * final_expand), n_classes, 1, bias=True),
        )

    def forward(self, feature_list):
        for i, m in enumerate(self.inputs):
            if i == 0:
                feat = m(feature_list[i])
            else:
                feat += m(feature_list[i])
        feat = self.middle(feat)
        return self.output(feat)

@MODELS.register_module()
class EVitSegHead(BaseHead):
    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int,
                 loss: ConfigType = dict(type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.head = SegHead(in_channels, head_width=64, head_depth=3, expand_ratio=4,
                            final_expand=4, n_classes=out_channels)
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        return self.head(feats)
    
    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False)
            )
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds
        
    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(output=to_numpy(pred_fields),
                                              target=to_numpy(gt_heatmaps),
                                              mask=to_numpy(keypoint_weights) > 0)
            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses
    
    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg