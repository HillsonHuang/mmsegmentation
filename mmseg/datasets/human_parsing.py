# coding=utf-8
import os.path as osp
import os
import glob
from .builder import DATASETS
from .custom import CustomDataset

'''@DATASETS.register_module()
class PascalContextDataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
               'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
               'window', 'wood')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    def __init__(self, split, **kwargs):
        super(PascalContextDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None'''


@DATASETS.register_module()
class Human_Parsing_Dataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    '''0 背景（Background）

    1 帽子（Hat）

    2 头发（Hair）

    3 手套（Glove）

    4 太阳镜（Sunglasses）

    5 上衣（UpperClothes） 

    6 裙子（Dress）

    7 外套（Coat）

    8 短袜（Socks）

    9 裤子（Pants）

    10 连身长裤（Jumpsuits）

    11 围巾（Scarf）

    12 短裙（Skirt） 

    13 脸（Face）

    14 左臂（Left-arm）

    15 右臂（Right-arm）

    16 左腿（Left-leg）

    17 右腿（Right-leg）

    18 左鞋（Left-shoe）

    19 右鞋（Right-shoe）'''
    CLASSES = ('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes',
               'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
               'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe',
               )

    PALETTE = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
               [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
               [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
               [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
               ]

    def __init__(self,
                 auto_split=None,
                 auto_mode='train',
                 **kwargs):
        super(Human_Parsing_Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

        self.auto_split = auto_split
        self.auto_mode = auto_mode
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix,
                                               self.split,
                                               auto_split=self.auto_split,
                                               auto_mode=self.auto_mode
                                               )

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split, *args, **kwargs):
        auto_split = kwargs['auto_split']
        auto_mode = kwargs['auto_mode']
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        elif auto_split is not None:
            folder_num = auto_split[0]
            folder_idx = auto_split[1]
            imgs = sorted(os.listdir(img_dir))
            files = []
            for idx in range(0, len(imgs)):
                file, suffix = osp.splitext(imgs[idx])
                if suffix == seg_map_suffix:
                    files.append(file)
            tra_idx, val_idx = self.folder_split(folder_num=folder_num, folder=folder_idx, max_idx=len(files))
            if auto_mode == 'train':
                imgs_idx = tra_idx
            else:
                imgs_idx = val_idx

            for img_idx in imgs_idx:
                img_name = files[img_idx]
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            # train_img = sorted(glob.glob(osp.join(img_dir,'*' + img_suffix)))
            # a = auto_split
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)


        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    @staticmethod
    def folder_split(folder_num=5, folder=1, max_idx=199):
        indexes = list(range(max_idx))
        val_len = int(max_idx / folder_num)
        val_idx = list(range((folder - 1) * val_len, folder * val_len))
        tra_idx = [i for i in indexes if i not in val_idx]
        return tra_idx, val_idx


