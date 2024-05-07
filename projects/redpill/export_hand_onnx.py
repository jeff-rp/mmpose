from mmengine.analysis import FlopAnalyzer
from mmpose.apis import init_model
import torch
from torch import nn
import sys
sys.path.append(r"D:\projects\HandClassification")
from model import HandClass
from mobileone import reparameterize_model

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_estimator = init_model(
            "projects/redpill/hand_mobileone-s1_dsntrle-192x192.py",
            "work_dirs/hand_mobileone-s1_dsntrle-192x192/best_AUC_epoch_90.pth",
            device='cuda',
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )
        self.hand_cls = HandClass('s0', inference_mode=True)
        self.hand_cls.load_state_dict(
            torch.load(r"D:\projects\HandClassification\checkpoint\s0v4\e260_accp0.954_acch0.867.pth")
        )
        self.hand_cls.cuda()

    def forward(self, x):
        y = self.pose_estimator.extract_feat(x)
        y = self.pose_estimator.head(y)[1]
        # x = x[:,:17]
        # x = self.pose_estimator.head.run_without_dsnt(x)
        y1 = self.hand_cls(x)
        return y, y1
    
model = Model().eval()
model.pose_estimator.backbone.switch_to_deploy()
model.hand_cls = reparameterize_model(model.hand_cls)
x = torch.rand((1, 3, 192, 192)).to('cuda')

flops = FlopAnalyzer(model, x)
print('GFLOPS:', flops.total() / 1024 / 1024 / 1024)

model_name = 'onnx/hand_mobileone-s1_dsntrle_cls_s0-192x192'
onnx_file = model_name + '.onnx'
torch.onnx.export(model, x, onnx_file, input_names=['image'],
                  output_names=['coordinates', 'presence', 'handness'])

import onnx
from onnxsim import simplify
model = onnx.load(onnx_file)
model_simp, check = simplify(model)
onnx.save(model_simp, onnx_file)

import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
onnx_fp16 = model_name + '_fp16.onnx'
model = onnxmltools.utils.load_model(onnx_file)
model_fp16 = convert_float_to_float16(model, keep_io_types=True)
onnxmltools.utils.save_model(model_fp16, onnx_fp16)