import numpy as np
import cupy as cp

from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from model.utils.nms import non_maximum_suppression

# 下面是ProposalTargetCreator代码：ProposalCreator产生2000个ROIS，
# 但是这些ROIS并不都用于训练，经过本ProposalTargetCreator的筛选产生
# 128个用于自身的训练

class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn
    # 输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、
    # 对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
    # 输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、
    # 128个gt_roi_label（128，1）
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape
        # 首先将2000个roi和m个bbox给concatenate了一下成为
        # 新的roi（2000+m，4）。
        roi = np.concatenate((roi, bbox), axis=0)
        # n_sample = 128,pos_ratio=0.5，round 对传入的数据进行四舍五入
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # 计算每一个roi与每一个bbox的iou
        iou = bbox_iou(roi, bbox)
        # 按行找到最大值，返回最大值对应的序号以及其真正的IOU。
        # 返回的是每个roi与哪个bbox的最大，以及最大的iou值
        gt_assignment = iou.argmax(axis=1)
        # 每个roi与对应bbox最大的iou
        max_iou = iou.max(axis=1)
        # 从1开始的类别序号，给每个类得到真正的label(将0-19变为1-20)
        gt_roi_label = label[gt_assignment] + 1

        # 同样的根据iou的最大值将正负样本找出来，pos_iou_thresh=0.5
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 需要保留的roi个数（满足大于pos_iou_thresh条件的roi与64之间较小的一个）
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        # 找出的样本数目过多就随机丢掉一些
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # neg_iou_thresh_hi=0.5，neg_iou_thresh_lo=0.0
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # 负样本label 设为0
        sample_roi = roi[keep_index]
        # 此时输出的128*4的sample_roi就可以去扔到 RoIHead网络里去进行分类
        # 与回归了。同样， RoIHead网络利用这sample_roi+featue为输入，输出
        # 是分类（21类）和回归（进一步微调bbox）的预测值，那么分类回归的groud 
        # truth就是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。
        # Compute offsets and scales to match sampled RoIs to the GTs.
        # 求这128个样本的groundtruth
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        # ProposalTargetCreator首次用到了真实的21个类的label,
        # 且该类最后对loc进行了归一化处理，所以预测时要进行均值方差处理
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label

# AnchorTargetCreator作用是生成训练要用的anchor(正负样本
# 各128个框的坐标和256个label（0或者1）)
# 利用每张图中bbox的真实标签来为所有任务分配ground truth
class AnchorTargetCreator(object):
    

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        # 特征图大小
        img_H, img_W = img_size
        # 一般对应20000个左右anchor
        n_anchor = len(anchor)
        # 将那些超出图片范围的anchor全部去掉,只保留位于图片内部的序号
        inside_index = _get_inside_index(anchor, img_H, img_W)
        # 保留位于图片内部的anchor
        anchor = anchor[inside_index]
        # 筛选出符合条件的正例128个负例128并给它们附上相应的label
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # 计算每一个anchor与对应bbox求得iou最大的bbox计算偏移
        # 量（注意这里是位于图片内部的每一个）
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # 将位于图片内部的框的label对应到所有生成的20000个框中
        # （label原本为所有在图片中的框的）
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        # 将回归的框对应到所有生成的20000个框中（label原本为
        # 所有在图片中的框的）
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label
    # 下面为调用的_creat_label（） 函数
    def _create_label(self, inside_index, anchor, bbox):
        # inside_index为所有在图片范围内的anchor序号
        label = np.empty((len(inside_index),), dtype=np.int32)
        # #全部填充-1
        label.fill(-1)
        # 调用_calc_ious（）函数得到每个anchor与哪个bbox的iou最大
        # 以及这个iou值、每个bbox与哪个anchor的iou最大(需要体会从
        # 行和列取最大值的区别)
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # #把每个anchor与对应的框求得的iou值与负样本阈值比较，若小
        # 于负样本阈值，则label设为0，pos_iou_thresh=0.7, 
        # neg_iou_thresh=0.3
        label[max_ious < self.neg_iou_thresh] = 0

        # 把与每个bbox求得iou值最大的anchor的label设为1
        label[gt_argmax_ious] = 1

        # 把每个anchor与对应的框求得的iou值与正样本阈值比较，
        # 若大于正样本阈值，则label设为1
        label[max_ious >= self.pos_iou_thresh] = 1

        # 按照比例计算出正样本数量，pos_ratio=0.5，n_sample=256
        n_pos = int(self.pos_ratio * self.n_sample)
        # 得到所有正样本的索引
        pos_index = np.where(label == 1)[0]
        # 如果选取出来的正样本数多于预设定的正样本数，则随机抛弃，将那些抛弃的样本的label设为-1
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 设定的负样本的数量
        n_neg = self.n_sample - np.sum(label == 1)
        # 负样本的索引
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            # 随机选择不要的负样本，个数为len(neg_index)-neg_index，label值设为-1
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
    # _calc_ious函数
    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        # 调用bbox_iou函数计算anchor与bbox的IOU， ious：（N,K），
        # N为anchor中第N个，K为bbox中第K个，N大概有15000个
        ious = bbox_iou(anchor, bbox)
        # 1代表行，0代表列
        argmax_ious = ious.argmax(axis=1)
        # 求出每个anchor与哪个bbox的iou最大，以及最大值，max_ious:[1,N]
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        # 求出每个bbox与哪个anchor的iou最大，以及最大值,gt_max_ious:[1,K]
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 然后返回最大iou的索引（每个bbox与哪个anchor的iou最大),有K个
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

# 将那些超出图片范围的anchor全部去掉,只保留位于图片内部的序号
def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

# 下面是ProposalCreator的代码： 这部分的操作不需要进行反向传播
# 因此可以利用numpy/tensor实现
class ProposalCreator:
    # 对于每张图片，利用它的feature map，计算（H/16）x(W/16)x9(大概20000)
    # 个anchor属于前景的概率，然后从中选取概率较大的12000张，利用位置回归参
    # 数，修正这12000个anchor的位置， 利用非极大值抑制，选出2000个ROIS以及
    # 对应的位置参数。
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    # 这里的loc和score是经过region_proposal_network中
    # 经过1x1卷积分类和回归得到的。
    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms #12000
            n_post_nms = self.n_train_post_nms #经过NMS后有2000个
        else:
            n_pre_nms = self.n_test_pre_nms #6000
            n_post_nms = self.n_test_post_nms #经过NMS后有300个

        # 将bbox转换为近似groudtruth的anchor(即rois)
        roi = loc2bbox(anchor, loc)

        # slice表示切片操作
        # 裁剪将rois的ymin,ymax限定在[0,H]
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        # 裁剪将rois的xmin,xmax限定在[0,W]
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale #16
        # rois的宽
        hs = roi[:, 2] - roi[:, 0]
        # rois的高
        ws = roi[:, 3] - roi[:, 1]
        # 确保rois的长宽大于最小阈值
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        # 对剩下的ROIs进行打分（根据region_proposal_network中rois的预测前景概率）
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        # 将score拉伸并逆序（从高到低）排序
        order = score.ravel().argsort()[::-1]
        # train时从20000中取前12000个rois，test取前6000个
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        # #（具体需要看NMS的原理以及输入参数的作用）调用非极大值抑制函数，
        # 将重复的抑制掉，就可以将筛选后ROIS进行返回。经过NMS处理后Train
        # 数据集得到2000个框，Test数据集得到300个框
        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        # 取出最终的2000或300个rois
        roi = roi[keep]
        return roi
