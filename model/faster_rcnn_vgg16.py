from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    # 是否使用Caffe下载下来的预训练模型
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            # 加载参数信息
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    # 加载预训练模型vgg16的conv5_3之前的部分
    features = list(model.features)[:30]

    classifier = model.classifier
    # 分类部分放到一个list里面
    classifier = list(classifier)
    # 删除输出分类结果层
    del classifier[6]
    # 删除两个dropout
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 冻结vgg16前2个stage,不进行反向传播
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    # 拆分为特征提取网络和分类网络
    return nn.Sequential(*features), classifier


# 分别对特征VGG16的特征提取部分、分类部分、RPN网络、
# VGG16RoIHead网络进行了实例化
class FasterRCNNVGG16(FasterRCNN):
    # vgg16通过5个stage下采样16倍
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16
    # 总类别数为20类，三种尺度三种比例的anchor
    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        
        # conv5_3及之前的部分，分类器
        extractor, classifier = decom_vgg16()

        # 返回rpn_locs, rpn_scores, rois, roi_indices, anchor
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )
        # 下面讲
        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        # vgg16中的最后两个全连接层
        self.classifier = classifier 
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        # 全连接层权重初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # 加上背景21类
        self.n_class = n_class
        # 7x7
        self.roi_size = roi_size
        # 1/16
        self.spatial_scale = spatial_scale
        # 将大小不同的roi变成大小一致，得到pooling后的特征，
        # 大小为[300, 512, 7, 7]。利用Cupy实现在线编译的
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
 
        # in case roi_indices is  ndarray
        # 前面解释过这里的roi_indices其实是多余的，因为batch_size一直为1
        roi_indices = at.totensor(roi_indices).float() #ndarray->tensor
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # 把tensor变成在内存中连续分布的形式
        # contiguous：view只能用在contiguous的variable上。
        # 如果在view之前用了transpose, permute等，需要用
        # contiguous()来返回一个contiguous copy。 
        indices_and_rois =  xy_indices_and_rois.contiguous()
        # 接下来分析roi_module.py中的RoI（）
        pool = self.roi(x, indices_and_rois)
        # flat操作
        pool = pool.view(pool.size(0), -1)
        # decom_vgg16（）得到的calssifier,得到4096
        fc7 = self.classifier(pool)
        # （4096->84）
        roi_cls_locs = self.cls_loc(fc7)
        # （4096->21）
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
