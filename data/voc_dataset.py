import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image


class VOCBboxDataset:
    
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        # id_list_file为split.txt，split为'trainval'或者'test'
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        # id_为每个样本文件名
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        # 写到/VOC2007/的路径
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        # 20类
        self.label_names = VOC_BBOX_LABEL_NAMES

    # trainval.txt有5011个，test.txt有210个
    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        #读入xml标签文件
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        #解析xml文件
        for obj in anno.findall('object'):
            # 标为difficult的目标在测试评估中一般会被忽略
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            #xml文件中包含object name和difficult(0或者1,0代表容易检测)
            difficult.append(int(obj.find('difficult').text))
            # bndbox（xmin,ymin,xmax,ymax),表示框左下角和右上角坐标
            bndbox_anno = obj.find('bndbox')
            # #让坐标基于（0,0）
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            # 框中object name
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        # 所有object的bbox坐标存在列表里
        bbox = np.stack(bbox).astype(np.float32)
        # 所有object的label存在列表里
        label = np.stack(label).astype(np.int32)
        # PyTorch 不支持 np.bool，所以这里转换为uint8
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  

        # 根据图片编号在/JPEGImages/取图片
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        # 如果color=True，则转换为RGB图
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    # 一般如果想使用索引访问元素时，就可以在类中定义这个方法（__getitem__(self, key) )
    __getitem__ = get_example

# 类别和名字对应的列表
VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
