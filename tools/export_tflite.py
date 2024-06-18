from mmpose.apis import init_model
import torch
from torch import nn
from tinynn.converter import TFLiteConverter

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_estimator = init_model(
            "projects/redpill/body_hardcorenas-c_lhr-256x256.py",
            "work_dirs/body_hardcorenas-c_lhr-256x256/best_coco_AP_epoch_210.pth",
            device='cuda',
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )

    def forward(self, x):
        x = self.pose_estimator.extract_feat(x)
        x = self.pose_estimator.head(x)
        return x

model = Model().eval().cpu()
# model.pose_estimator.backbone.switch_to_deploy()

x = torch.rand((1, 3, 256, 256))
model_name = 'onnx/body_hardcorenas-c_lhr-256x256'

tflite_file = model_name + '.tflite'
# torch.backends.quantized.engine = 'qnnpack'
converter = TFLiteConverter(model, x, tflite_file, float16_quantization=True)
converter.convert()