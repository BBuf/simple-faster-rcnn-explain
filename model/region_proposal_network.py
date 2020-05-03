import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # 首先生成上述以（0，0）为中心的9个base anchor
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
       # x的尺寸为(batch_size，512,H/16,W/16），其中H，W分别为原图的高和宽
        # x为feature map，n为batch_size,此版本代码为1. hh，ww即为宽高
        n, _, hh, ww = x.shape
        # 在9个base_anchor基础上生成hh*ww*9个anchor，对应到原图坐标
        # feat_stride=16 ，因为是经4次pool后提到的特征，故feature map较
        # 原图缩小了16倍
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),/
            self.feat_stride, hh, ww)
        
        # （hh * ww * 9）/hh*ww = 9 
        n_anchor = anchor.shape[0] // (hh * ww) 
        # 512个3x3卷积(512, H/16,W/16)
        h = F.relu(self.conv1(x))
        # n_anchor（9）* 4个1x1卷积，回归坐标偏移量。（9*4，hh,ww)
        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        # 转换为（n，hh，ww，9*4）后变为（n，hh*ww*9，4）
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # n_anchor（9）*2个1x1卷积，回归类别。（9*2，hh,ww）
        rpn_scores = self.score(h)
        # 转换为（n，hh，ww，9*2）
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # 计算{Softmax}(x_{i}) = \{exp(x_i)}{\sum_j exp(x_j)}
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # 得到前景的分类概率
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        # 得到所有anchor的前景分类概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # 得到每一张feature map上所有anchor的网络输出值
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        # n为batch_size数
        for i in range(n):
            # 调用ProposalCreator函数， rpn_locs维度（hh*ww*9，4）
            # ，rpn_fg_scores维度为（hh*ww*9），anchor的维度为
            # （hh*ww*9，4）， img_size的维度为（3，H，W），H和W是
            # 经过数据预处理后的。计算（H/16）x(W/16)x9(大概20000)
            # 个anchor属于前景的概率，取前12000个并经过NMS得到2000个
            # 近似目标框G^的坐标。roi的维度为(2000,4)
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            # rois为所有batch_size的roi
            rois.append(roi)
            roi_indices.append(batch_index)
        # 按行拼接（即没有batch_size的区分，每一个[]里都是一个anchor的四个坐标）
        rois = np.concatenate(rois, axis=0)
        # 这个 roi_indices在此代码中是多余的，因为我们实现的是batch_siae=1的
        # 网络，一个batch只会输入一张图象。如果多张图像的话就需要存储索引以找到
        # 对应图像的roi
        roi_indices = np.concatenate(roi_indices, axis=0)
        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        # rois的维度为（2000,4），roi_indices用不到，
        # anchor的维度为（hh*ww*9，4）
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


# 利用anchor_base生成所有对应feature map的anchor
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    # 纵向偏移量（0，16，32，...）
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    # 横向偏移量（0，16，32，...）
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    # shift_x = [[0，16，32，..],[0，16，32，..],[0，16，32，..]...],
    # shift_y = [[0，0，0，..],[16，16，16，..],[32，32，32，..]...],
    # 就是形成了一个纵横向偏移量的矩阵，也就是特征图的每一点都能够通过这个
    # 矩阵找到映射在原图中的具体位置！
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    # 经过刚才的变化，其实大X,Y的元素个数已经相同，看矩阵的结构也能看出，
    # 矩阵大小是相同的，X.ravel()之后变成一行，此时shift_x,shift_y的元
    # 素个数是相同的，都等于特征图的长宽的乘积(像素点个数)，不同的是此时
    # 的shift_x里面装得是横向看的x的一行一行的偏移坐标，而此时的y里面装
    # 的是对应的纵向的偏移坐标！
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    # A=9
    A = anchor_base.shape[0]
    # 读取特征图中元素的总个数
    K = shift.shape[0]
    #用基础的9个anchor的坐标分别和偏移量相加，最后得出了所有的anchor的坐标，
    # 四列可以堪称是左上角的坐标和右下角的坐标加偏移量的同步执行，飞速的从
    # 上往下捋一遍，所有的anchor就都出来了！一共K个特征点，每一个有A(9)个
    # 基本的anchor，所以最后reshape((K*A),4)的形式，也就得到了最后的所有
    # 的anchor左下角和右上角坐标.          
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


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
