# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class FlowerDataset(CustomDataset):
    """
    Flower Dataset
    """
    CLASSES = ('Backbone', 'Flower')
    PALETTE = [[0, 0, 128], [128, 0, 0]]

    def __init__(self, **kwargs):
        super(FlowerDataset, self).__init__(
            # classes=['background', 'Apple'],
            img_suffix='.png',
            # split='split.txt',
            seg_map_suffix='.png',
            ignore_index=None,

            # 开启后，原本为0的标注设置为255,被忽略，不参与损失计算，所有标签值-1
            # reduce_zero_label=True,
            **kwargs),
        self.custom_classes = True
        self.label_map = dict({255: 1})
        assert self.file_client.exists(self.img_dir)
