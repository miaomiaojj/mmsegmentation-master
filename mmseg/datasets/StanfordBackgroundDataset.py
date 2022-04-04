import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class StanfordBackgroundDataset(CustomDataset):
    CLASSES = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
    PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
    def __init__(self, **kwargs):
        super(StanfordBackgroundDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)