from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    # 预测框的位置，预测框的类别和分数
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    # 真实框的位置，类别，是否为明显目标
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # 一个for循环，从 enumerate(dataloader)里面依次读取数据，
    # 读取的内容是: imgs图片，sizes尺寸，gt_boxes真实框的位置
    #  gt_labels真实框的类别以及gt_difficults
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # 用faster_rcnn.predict(imgs,[sizes]) 得出预测的pred_boxes_,
        # pred_labels_,pred_scores_预测框位置，预测框标记以及预测框
        # 的分数等等
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    # 将pred_bbox,pred_label,pred_score ,gt_bbox,gt_label,gt_difficult
    # 预测和真实的值全部依次添加到开始定义好的列表里面去，如果迭代次数等于测
    # 试test_num，那么就跳出循环！调用 eval_detection_voc函数，接收上述的
    # 六个列表参数，完成预测水平的评估！得到预测的结果
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    # opt._parse(kwargs)#将调用函数时候附加的参数用，
    # config.py文件里面的opt._parse()进行解释，然后
    # 获取其数据存储的路径，之后放到Dataset里面！
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    # #Dataset完成的任务见第二次推文数据预处理部分，
    # 这里简单解释一下，就是用VOCBboxDataset作为数据
    # 集，然后依次从样例数据库中读取图片出来，还调用了
    # Transform(object)函数，完成图像的调整和随机翻转工作
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    # 将数据装载到dataloader中，shuffle=True允许数据打乱排序，
    # num_workers是设置数据分为几批处理，同样的将测试数据集也
    # 进行同样的处理，然后装载到test_dataloader中
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    # 定义faster_rcnn=FasterRCNNVGG16()训练模型
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')

    # 设置trainer = FasterRCNNTrainer(faster_rcnn).cuda()将
    # FasterRCNNVGG16作为fasterrcnn的模型送入到FasterRCNNTrainer
    # 中并设置好GPU加速
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    # 用一个for循环开始训练过程，而训练迭代的次数
    # opt.epoch=14也在config.py文件中预先定义好，属于超参数
    for epoch in range(opt.epoch):
        # 首先在可视化界面重设所有数据
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            # 然后从训练数据中枚举dataloader,设置好缩放范围，
            # 将img,bbox,label,scale全部设置为可gpu加速
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # 调用trainer.py中的函数trainer.train_step
            # (img,bbox,label,scale)进行一次参数迭代优化过程
            trainer.train_step(img, bbox, label, scale)

            # 判断数据读取次数是否能够整除plot_every
            # (是否达到了画图次数)，如果达到判断debug_file是否存在，
            # 用ipdb工具设置断点，调用trainer中的trainer.vis.
            # plot_many(trainer.get_meter_data())将训练数据读取并
            # 上传完成可视化
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                # 将每次迭代读取的图片用dataset文件里面的inverse_normalize()
                # 函数进行预处理，将处理后的图片调用Visdom_bbox可视化 
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                # 调用faster_rcnn的predict函数进行预测，
                # 预测的结果保留在以_下划线开头的对象里面
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                # 利用同样的方法将原始图片以及边框类别的
                # 预测结果同样在可视化工具中显示出来
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                # 调用trainer.vis.text将rpn_cm也就是
                # RPN网络的混淆矩阵在可视化工具中显示出来
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # 可视化ROI head的混淆矩阵
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        # 调用eval函数计算map等指标
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # 可视化map
        trainer.vis.plot('test_map', eval_result['map'])
        # 设置学习的learning rate
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        # 将损失学习率以及map等信息及时显示更新
        trainer.vis.log(log_info)
        # 用if判断语句永远保存效果最好的map
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            # if判断语句如果学习的epoch达到了9就将学习率*0.1
            # 变成原来的十分之一
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        # 判断epoch==13结束训练验证过程
        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
