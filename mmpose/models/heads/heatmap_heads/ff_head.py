import torch
from mmengine.structures import PixelData
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np
from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from typing import Optional, Sequence, Tuple, Union

OptIntSeq = Optional[Sequence[int]]

BN_MOMENTUM = 0.1
up_kwargs = {"mode": "bilinear", "align_corners": True}
# up_kwargs = {"mode": "nearest"}
relu_inplace = True

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, activation=nn.ReLU,
                 *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        layers = [
            nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_chan, momentum=BN_MOMENTUM)
        ]
        if activation:
            layers.append(activation(inplace=relu_inplace))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class AdapterConv(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=[64, 128, 256, 512]):
        super(AdapterConv, self).__init__()
        assert len(in_channels) == len(
            out_channels
        ), "Number of input and output branches should match"
        self.adapter_conv = nn.ModuleList()

        for k in range(len(in_channels)):
            self.adapter_conv.append(
                ConvBNReLU(in_channels[k], out_channels[k], ks=1, stride=1, padding=0),
            )

    def forward(self, x):
        out = []
        for k in range(len(self.adapter_conv)):
            out.append(self.adapter_conv[k](x[k]))
        return out

class UpsampleCat(nn.Module):
    def __init__(self, upsample_kwargs=up_kwargs):
        super(UpsampleCat, self).__init__()
        self._up_kwargs = upsample_kwargs

    def forward(self, x):
        """Upsample and concatenate feature maps."""
        assert isinstance(x, list) or isinstance(x, tuple)
        x0 = x[0]
        _, _, H, W = x0.size()
        for i in range(1, len(x)):
            x0 = torch.cat([x0, F.interpolate(x[i], (H, W), **self._up_kwargs)], dim=1)
        return x0
    
class UpBranch(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=[128, 128, 128, 128],
                 upsample_kwargs=up_kwargs):
        super(UpBranch, self).__init__()

        self._up_kwargs = upsample_kwargs

        self.fam_32_sm = ConvBNReLU(in_channels[3], out_channels[3], ks=3, stride=1, padding=1)
        self.fam_32_up = ConvBNReLU(in_channels[3], in_channels[2], ks=1, stride=1, padding=0)
        self.fam_16_sm = ConvBNReLU(in_channels[2], out_channels[2], ks=3, stride=1, padding=1)
        self.fam_16_up = ConvBNReLU(in_channels[2], in_channels[1], ks=1, stride=1, padding=0)
        self.fam_8_sm = ConvBNReLU(in_channels[1], out_channels[1], ks=3, stride=1, padding=1)
        self.fam_8_up = ConvBNReLU(in_channels[1], in_channels[0], ks=1, stride=1, padding=0)
        self.fam_4 = ConvBNReLU(in_channels[0], out_channels[0], ks=3, stride=1, padding=1)

        self.high_level_ch = sum(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        feat4, feat8, feat16, feat32 = x

        smfeat_32 = self.fam_32_sm(feat32)
        upfeat_32 = self.fam_32_up(feat32)

        _, _, H, W = feat16.size()
        x = F.interpolate(upfeat_32, (H, W), **self._up_kwargs) + feat16
        smfeat_16 = self.fam_16_sm(x)
        upfeat_16 = self.fam_16_up(x)

        _, _, H, W = feat8.size()
        x = F.interpolate(upfeat_16, (H, W), **self._up_kwargs) + feat8
        smfeat_8 = self.fam_8_sm(x)
        upfeat_8 = self.fam_8_up(x)

        _, _, H, W = feat4.size()
        smfeat_4 = self.fam_4(F.interpolate(upfeat_8, (H, W), **self._up_kwargs) + feat4)

        return smfeat_4, smfeat_8, smfeat_16, smfeat_32
    
class FFNetUpHead(nn.Module):
    def __init__(self, in_chans, use_adapter_conv, up_kwargs, head_type="B", task="segmentation_A",
                 num_classes=17):
        super(FFNetUpHead, self).__init__()

        layers = []
        if head_type == "A":
            base_chans = [64, 128, 256, 512]
        elif head_type == "B":
            base_chans = [64, 128, 128, 256]
        elif head_type == "C":
            base_chans = [128, 128, 128, 128]

        if use_adapter_conv:
            layers.append(AdapterConv(in_chans, base_chans))
            in_chans = base_chans[:]

        if head_type == "A":
            layers.append(UpBranch(in_chans, upsample_kwargs=up_kwargs))
        elif head_type == "B":
            layers.append(UpBranch(in_chans, [96, 96, 64, 32], upsample_kwargs=up_kwargs))
        elif head_type == "C":
            layers.append(UpBranch(in_chans, [128, 16, 16, 16], upsample_kwargs=up_kwargs))
        else:
            raise ValueError(f"Unknown FFNetUpHead type {head_type}")

        self.num_features = layers[-1].high_level_ch
        self.num_multi_scale_features = layers[-1].out_channels

        if task.startswith("segmentation"):
            layers.append(UpsampleCat(up_kwargs))

            # Gets single scale input
            if "_C" in task:
                mid_feat = 128
                layers.append(
                    SegmentationHead_NoSigmoid_1x1(
                        self.num_features,
                        mid_feat,
                        num_outputs=num_classes,
                    )
                )
            elif "_B" in task:
                mid_feat = 256
                layers.append(
                    SegmentationHead_NoSigmoid_3x3(
                        self.num_features,
                        mid_feat,
                        num_outputs=num_classes,
                    )
                )
            elif "_A" in task:
                mid_feat = 512
                layers.append(
                    SegmentationHead_NoSigmoid_1x1(
                        self.num_features,
                        mid_feat,
                        num_outputs=num_classes,
                    )
                )
            else:
                raise ValueError(f"Unknown Segmentation Head {task}")
        elif task == "classification":
            # Gets multi scale input
            layers.append(
                ClassificationHead(
                    self.num_multi_scale_features,
                    [128, 256, 512, 1024],
                    num_outputs=num_classes,
                    dropout_rate=None,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SimpleBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(SimpleBottleneckBlock, self).__init__()
        bn_mom = 0.1
        bn_eps = 1e-5

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom),
            )

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ClassificationHead(nn.Module):
    def __init__(
        self,
        pre_head_channels,
        # head_channels=[128, 256, 512, 1024],
        head_channels=[64, 128, 256, 512],
        num_outputs=1,
        dropout_rate=None,
    ):
        super(ClassificationHead, self).__init__()

        self.dropout_rate = dropout_rate
        bn_mom = 0.1
        bn_eps = 1e-5
        head_block_type = SimpleBottleneckBlock
        head_expansion = 4

        expansion_layers = []
        for i, pre_head_channel in enumerate(pre_head_channels):
            expansion_layer = head_block_type(
                pre_head_channel,
                int(head_channels[i] / head_expansion),
            )
            expansion_layers.append(expansion_layer)
        self.expansion_layers = nn.ModuleList(expansion_layers)

        # downsampling modules
        downsampling_layers = []
        for i in range(len(pre_head_channels) - 1):
            input_channels = head_channels[i]
            output_channels = head_channels[i + 1]

            downsampling_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(output_channels, momentum=bn_mom),
                nn.ReLU(),
            )

            downsampling_layers.append(downsampling_layer)
        self.downsampling_layers = nn.ModuleList(downsampling_layers)

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[-1],
                # out_channels=2048,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.BatchNorm2d(2048, momentum=bn_mom),
            nn.BatchNorm2d(1024, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(
            1024, #2048,
            num_outputs,
        )

    def forward(self, x):

        next_x = self.expansion_layers[0](x[0])
        for i in range(len(self.downsampling_layers)):
            next_x = self.expansion_layers[i + 1](x[i + 1]) + self.downsampling_layers[
                i
            ](next_x)
        x = next_x

        x = self.final_layer(x)
        x = self.adaptive_avg_pool(x).squeeze()

        if self.dropout_rate:
            x = torch.nn.functional.dropout(
                x, p=self._model_config.dropout_rate, training=self.training
            )

        x = self.classifier(x)
        return x

class SegmentationHead_NoSigmoid_3x3(nn.Module):
    def __init__(self, backbone_channels, mid_channels=256, kernel_size=3, num_outputs=1):
        super(SegmentationHead_NoSigmoid_3x3, self).__init__()
        last_inp_channels = backbone_channels
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=num_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
        )

    def forward(self, x):
        x = self.last_layer(x)
        return x

class SegmentationHead_NoSigmoid_1x1(nn.Module):
    def __init__(self, backbone_channels, mid_channels=512, kernel_size=3, num_outputs=1):
        super(SegmentationHead_NoSigmoid_1x1, self).__init__()
        last_inp_channels = backbone_channels
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=num_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.last_layer(x)
        return x

@MODELS.register_module()
class FFHead(BaseHead):
    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 arch_type: str = "C",
                 up_linear: bool = True,
                 use_adapter_convs = True,
                 loss: ConfigType = dict(type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        up_kwargs = {"mode": "bilinear", "align_corners": True} if up_linear else {"mode": "nearest"}
        head_type = arch_type
        task = "segmentation_" + head_type
        # task = "classification"
        self.head = FFNetUpHead(in_channels, use_adapter_convs, up_kwargs, head_type, task, out_channels)
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
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
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
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0
            )
            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.01),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg