from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        # 继承父模块的初始化
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        # 下面2个参数是在_faster_rcnn_loc_loss调用用来计算位置损失函数用到的超参数
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        # 用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，也就是
        # 为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
        self.anchor_target_creator = AnchorTargetCreator()
        # AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标
        # （或称ground truth），只在训练阶段用到，ProposalCreator是RPN为Fast
        #  R-CNN生成RoIs，在训练和测试阶段都会用到。所以测试阶段直接输进来300
        # 个RoIs，而训练阶段会有AnchorTargetCreator的再次干预
        self.proposal_target_creator = ProposalTargetCreator()
        # (0., 0., 0., 0.)
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        # (0.1, 0.1, 0.2, 0.2)
        self.loc_normalize_std = faster_rcnn.loc_normalize_std
        # SGD
        self.optimizer = self.faster_rcnn.get_optimizer()
        # 可视化，vis_tool.py
        self.vis = Visualizer(env=opt.env)

        # 混淆矩阵，就是验证预测值与真实值精确度的矩阵ConfusionMeter
        # (2)括号里的参数指的是类别数
        self.rpn_cm = ConfusionMeter(2)
        # roi的类别有21种（20个object类+1个background）
        self.roi_cm = ConfusionMeter(21)
        # 平均损失
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        # 获取batch个数
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        # （n,c,hh,ww）
        img_size = (H, W)

        # vgg16 conv5_3之前的部分提取图片的特征
        features = self.faster_rcnn.extractor(imgs)

        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        #  rois的维度为（2000,4），roi_indices用不到，anchor的维度为
        # （hh*ww*9，4），H和W是经过数据预处理后的。计算（H/16）x(W/16)x9
        # (大概20000)个anchor属于前景的概率，取前12000个并经过NMS得到2000个
        # 近似目标框G^的坐标。roi的维度为(2000,4)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # bbox维度(N, R, 4)
        bbox = bboxes[0]
        # labels维度为（N，R）
        label = labels[0]
        #hh*ww*9
        rpn_score = rpn_scores[0]
        # hh*ww*9
        rpn_loc = rpn_locs[0]
        # (2000,4)
        roi = rois

        # Sample RoIs and forward
        # 调用proposal_target_creator函数生成sample roi（128，4）、
        # gt_roi_loc（128，4）、gt_roi_label（128，1），RoIHead网络
        # 利用这sample_roi+featue为输入，输出是分类（21类）和回归
        # （进一步微调bbox）的预测值，那么分类回归的groud truth就
        # 是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        # roi回归输出的是128*84和128*21，然而真实位置参数是128*4和真实标签128*1
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # 输入20000个anchor和bbox，调用anchor_target_creator函数得到
        # 2000个anchor与bbox的偏移量与label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        # 下面分析_fast_rcnn_loc_loss函数。rpn_loc为rpn网络回归出来的偏移量
        # （20000个），gt_rpn_loc为anchor_target_creator函数得到2000个anchor
        # 与bbox的偏移量，rpn_sigma=1.
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # rpn_score为rpn网络得到的（20000个）与anchor_target_creator
        # 得到的2000个label求交叉熵损失
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1] #不计算背景类
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        # roi_cls_loc为VGG16RoIHead的输出（128*84）， n_sample=128
        n_sample = roi_cls_loc.shape[0]
        # roi_cls_loc=（128,21,4）
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        # proposal_target_creator()生成的128个proposal与bbox求得的偏移量
        # dx,dy,dw,dh
        gt_roi_label = at.totensor(gt_roi_label).long()
        # 128个标签
        gt_roi_loc = at.totensor(gt_roi_loc)
        # 采用smooth_l1_loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        # 求交叉熵损失
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())
        # 四个loss加起来
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)
    # 整个函数实际上就是进行了一次参数的优化过程，首先`self.optimizer.zero_grad()`将梯度数据全部清零，
    # 然后利用刚刚介绍`self.forward(imgs,bboxes,labels,scales)`函数将所有的损失计算出来，接着进行
    # 依次`losses.total_loss.backward()`反向传播计算梯度，`self.optimizer.step()`进行一次参数
    # 更新过程，`self.update_meters(losses)`就是将所有损失的数据更新到可视化界面上,最后将`losses`返回
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses
    # 模型保存
    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path
    # 模型加载
    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

# 输入分别为rpn回归框的偏移量和anchor与bbox的偏移量以及label
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    # sigma设置为1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    # 除去背景类
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
