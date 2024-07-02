from mmpose.apis import init_model
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_estimator = init_model(
            "projects/redpill/body_mobilenetv4-s_seg-256x256.py",
            "work_dirs/body_mobilenetv4-s_seg-256x256/best_coco_AP_epoch_230.pth",
            device='cuda',
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )

    def forward(self, x):
        x = self.pose_estimator.extract_feat(x)
        x = self.pose_estimator.head(x)
        return x

model = Model().eval()
# model.pose_estimator.backbone.switch_to_deploy()
x = torch.rand((1, 3, 256, 256)).to('cuda')
model_name = 'onnx/body_mobilenetv4-s_seg-256x256'
onnx_file = model_name + '.onnx'
torch.onnx.export(model, x, onnx_file, input_names=['image'], output_names=['heatmap'])
# torch.onnx.export(model, x, onnx_file, input_names=['image'], output_names=['heatmap', 'visibility'])
# torch.onnx.export(model, x, onnx_file, input_names=['image'], output_names=['output_x', 'output_y'])

import onnx
from onnxsim import simplify
model = onnx.load(onnx_file)
model_simp, check = simplify(model)
onnx.save(model_simp, onnx_file)

# from onnxconverter_common import float16
# onnx_fp16 = model_name + '_fp16.onnx'
# model = onnx.load(onnx_file)
# model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
# onnx.save(model_fp16, onnx_fp16)