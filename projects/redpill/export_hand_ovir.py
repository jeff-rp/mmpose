from mmengine.analysis import FlopAnalyzer
from mmpose.apis import init_model
import torch
from torch import nn
import openvino as ov
import sys
sys.path.append(r"D:\projects\HandClassification")
from model import HandClass
from mobileone import reparameterize_model

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_estimator = init_model(
            "work_dirs/hand_mobileone-s1_rle-128x128/hand_mobileone-s1_rle-128x128.py",
            "work_dirs/hand_mobileone-s1_rle-128x128/best_AUC_epoch_140.pth",
            device='cuda',
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )
        self.hand_cls = HandClass('s0', inference_mode=True)
        self.hand_cls.load_state_dict(
            torch.load(r"D:\projects\HandClassification\checkpoint\s0v4-128\e80_accp0.956_acch0.859.pth")
        )
        self.hand_cls.cuda()

    def forward(self, image):
        y = self.pose_estimator.extract_feat(image)
        coordinates = self.pose_estimator.head(y)
        presence, handness = self.hand_cls(image)

        return coordinates, presence, handness
    
model = Model().eval()
model.pose_estimator.backbone.switch_to_deploy()
model.hand_cls = reparameterize_model(model.hand_cls)
image = torch.rand((2, 3, 128, 128)).to('cuda')

inputs = { "image": ([2, 3, 128, 128], ov.Type.f32) }
ov_model = ov.convert_model(model, input=inputs, example_input=image)
model_name = 'onnx/hand_mobileone-s1_rle_cls_s0-128x128_b2'
ir_file = model_name + '.xml'
ov.save_model(ov_model, ir_file, compress_to_fp16=True)