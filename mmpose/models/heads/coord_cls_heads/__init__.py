# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .rtmw_head import RTMWHead
from .simcc_head import SimCCHead
from .evitcc_head import EVitCCHead

__all__ = ['SimCCHead', 'RTMCCHead', 'RTMWHead', 'EVitCCHead']
