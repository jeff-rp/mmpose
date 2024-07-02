import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule

class SE(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, int(channels / ratio), 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(channels / ratio), channels, 1),
            nn.Hardsigmoid(inplace=True),
        )
        # self.conv1 = ConvModule(
        #     in_channels=channels,
        #     out_channels=int(channels / ratio),
        #     kernel_size=1,
        #     stride=1,
        #     conv_cfg=conv_cfg,
        #     act_cfg=act_cfg[0]
        # )
        # self.conv2 = ConvModule(
        #     in_channels=int(channels / ratio),
        #     out_channels=channels,
        #     kernel_size=1,
        #     stride=1,
        #     conv_cfg=conv_cfg,
        #     act_cfg=act_cfg[1]
        # )

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class Spatial(nn.Module):
    def __init__(self, channels, conv_cfg=None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Hardsigmoid(inplace=True)
        )
        # self.conv1 = ConvModule(in_channels=channels, out_channels=1, kernel_size=1, stride=1,
        #                         conv_cfg=conv_cfg, act_cfg=act_cfg[0])
        # self.conv2 = ConvModule(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, 
        #                         conv_cfg=conv_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x * out

class Attention(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        
        self.se = SE(channels, ratio)
        self.sse = Spatial(channels)

    def forward(self, x):
        out1 = self.se(x)
        out2 = self.sse(x)
        return x + out1 + out2