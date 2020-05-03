from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

# 去正则化,img维度为[[B,G,R],H,W],因为caffe预训练模型输入为BGR 0-255图片，pytorch预训练模型采用RGB 0-1图片
def inverse_normalize(img):
    if opt.caffe_pretrain:
        # [122.7717, 115.9465, 102.9801]reshape为[3,1,1]与img维度相同就可以相加了，
        # pytorch_normalize之前有减均值预处理，现在还原回去。
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        # 将BGR转换为RGB图片（python [::-1]为逆序输出）
        return img[::-1, :, :]
    # pytorch_normalze中标准化为减均值除以标准差，现在乘以标准差加上均值还原回去，转换为0-255
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

# 采用pytorch预训练模型对图片预处理，函数输入的img为0-1
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    # #transforms.Normalize使用如下公式进行归一化
    # channel=（channel-mean）/std,转换为[-1,1]
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    # nddarry->Tensor
    img = normalize(t.from_numpy(img))
    return img.numpy()

# 采用caffe预训练模型时对输入图像进行标准化，函数输入的img为0-1
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    # RGB-BGR
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    # 转换为与img维度相同
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    # 减均值操作
    img = (img - mean).astype(np.float32, copy=True)
    return img

# 函数输入的img为0-255
def preprocess(img, min_size=600, max_size=1000):
    # 图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。
    # 对相应的bounding boxes 也也进行同等尺度的缩放。
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    # 选小的比例，这样长和宽都能放缩到规定的尺寸
    scale = min(scale1, scale2)
    img = img / 255.
    # resize到（H * scale, W * scale）大小，anti_aliasing为是否采用高斯滤波
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    #调用pytorch_normalze或者caffe_normalze对图像进行正则化
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        # 图像等比例缩放
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        # 得出缩放比因子
        scale = o_H / H
        # bbox按照与原图等比例缩放
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # 将图片进行随机水平翻转，没有进行垂直翻转
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        # 同样地将bbox进行与对应图片同样的水平翻转
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

# 训练集样本的生成
class Dataset:
    def __init__(self, opt):
        self.opt = opt
         # 实例化类
        self.db = VOCBboxDataset(opt.voc_data_dir)
        #实例化类
        self.tsf = Transform(opt.min_size, opt.max_size)
    # __ xxx__运行Dataset类时自动运行
    def __getitem__(self, idx):
        # 调用VOCBboxDataset中的get_example()从数据集存储路径中将img, bbox, label, difficult 一个个的获取出来
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        # 调用前面的Transform函数将图片,label进行最小值最大值放缩归一化，
        # 重新调整bboxes的大小，然后随机反转，最后将数据集返回
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

# 测试集样本的生成
class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        # 此处设置了use_difficult,
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
