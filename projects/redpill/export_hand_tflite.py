from mmpose.apis import init_model
import torch
from torch import nn
import sys
from tinynn.converter import TFLiteConverter
sys.path.append(r"D:\projects\HandClassification")
from model import HandClass
from mobileone import reparameterize_model

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_estimator = init_model(
            "work_dirs/hand_hgnetv2-b0_rle/hand_hgnetv2-b0_rle.py",
            "work_dirs/hand_hgnetv2-b0_rle/best_AUC_epoch_180.pth",
            device='cuda',
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )
        self.hand_cls = HandClass('s0', inference_mode=True)
        self.hand_cls.load_state_dict(
            torch.load(r"D:\projects\HandClassification\checkpoint\s0v4-128\e80_accp0.956_acch0.859.pth")
        )
        self.hand_cls.cuda()

    def forward(self, x):
        y = self.pose_estimator.extract_feat(x)
        y = self.pose_estimator.head(y) #[1]#dsnt heatmap
        # x = x[:,:17]
        # x = self.pose_estimator.head.run_without_dsnt(x)
        y1 = self.hand_cls(x)
        return y, y1
    
model = Model().eval().cpu()
# model.pose_estimator.backbone.switch_to_deploy()
model.hand_cls = reparameterize_model(model.hand_cls)

x = torch.rand((2, 3, 128, 128))
model_name = 'onnx/hand_hgnetv2-b0_rle_cls_s0-128x128_b2'

tflite_file = model_name + '.tflite'
converter = TFLiteConverter(model, x, tflite_file)
converter.convert()