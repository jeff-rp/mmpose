from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset

@DATASETS.register_module()
class Coco29Dataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/coco29.py')