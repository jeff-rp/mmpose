from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset

@DATASETS.register_module()
class Coco59Dataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/coco59.py')