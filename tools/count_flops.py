from mmengine.analysis import FlopAnalyzer
from mmpose.apis import init_model
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_estimator = init_model(
            "projects/redpill/ubody_mobileone-s1-ffc.py",
            None,
            device='cuda',
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )

    def forward(self, x):
        x = self.pose_estimator.extract_feat(x)
        x = self.pose_estimator.head(x)
        return x
    
model = Model().eval()
model.pose_estimator.backbone.switch_to_deploy()
x = torch.rand((1, 3, 256, 256)).to('cuda')
flops = FlopAnalyzer(model, x)
print('GFLOPS:', flops.total() / 1024 / 1024 / 1024)