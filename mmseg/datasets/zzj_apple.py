# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AppleDataset(CustomDataset):
    """
    Apple Dataset
    """
    CLASSES = ('Backbone', 'Apple')
    PALETTE = [[0, 0, 128], [128, 0, 0]]


    def __init__(self, **kwargs):
        super(AppleDataset, self).__init__(
            # classes=['background', 'Apple'],
            img_suffix='.png',
            seg_map_suffix='.png',
            # 开启后，原本为0的标注设置为255,被忽略，不参与损失计算，所有标签值-1
            # reduce_zero_label=True,
            **kwargs),
        self.custom_classes = True
        self.label_map = dict(zip([x for x in range(2, 150)], [1] * 150))
        assert self.file_client.exists(self.img_dir)
