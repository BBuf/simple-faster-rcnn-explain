import numpy as np
import numpy as xp

import six
from six import __init__

# 已知源bbox和位置偏差dx,dy,dh,dw，求目标框G
def loc2bbox(src_bbox, loc):
    
    # src_bbox：（R，4），R为bbox个数，4为左上角和右下角四个坐标
    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    #src_height为Ph,src_width为Pw，src_ctr_y为Py，src_ctr_x为Px
    src_height = src_bbox[:, 2] - src_bbox[:, 0]  #ymax-ymin
    src_width = src_bbox[:, 3] - src_bbox[:, 1] #xmax-xmin
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height #y0+0.5h
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width #x0+0.5w,计算出中心点坐标

    #python [start:stop:step] 
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # RCNN中提出的边框回归：寻找原始proposal与近似目标框G之间的映射关系，公式在上面
    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis] #ctr_y为Gy
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis] # ctr_x为Gx
    h = xp.exp(dh) * src_height[:, xp.newaxis] #h为Gh
    w = xp.exp(dw) * src_width[:, xp.newaxis] #w为Gw
    # 上面四行得到了回归后的目标框（Gx,Gy,Gh,Gw）

    # 由中心点转换为左上角和右下角坐标
    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

# 已知源框和目标框求出其位置偏差
def bbox2loc(src_bbox, dst_bbox):
    
    # 计算出源框中心点坐标
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    # 计算出目标框中心点坐标
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 求出最小的正数
    eps = xp.finfo(height.dtype).eps
    # 将height,width与其比较保证全部是非负
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    # 根据上面的公式二计算dx，dy，dh，dw
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    # np.vstack按照行的顺序把数组给堆叠起来
    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc

# 求两个bbox的相交的交并比
def bbox_iou(bbox_a, bbox_b):
    # 确保bbox第二维为bbox的四个坐标（ymin，xmin，ymax，xmax）
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    # l为交叉部分框左上角坐标最大值，为了利用numpy的广播性质，
    # bbox_a[:, None, :2]的shape是(N,1,2)，bbox_b[:, :2]
    # shape是(K,2),由numpy的广播性质，两个数组shape都变成(N,K,2)，
    # 也就是对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    # br为交叉部分框右下角坐标最小值
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    # 所有坐标轴上tl<br时，返回数组元素的乘积(y1max-yimin)X(x1max-x1min)，
    # bboxa与bboxb相交区域的面积
    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 计算bboxa的面积
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # 计算bboxb的面积
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    # 计算IOU
    return area_i / (area_a[:, None] + area_b - area_i)


def __test():
    pass


if __name__ == '__main__':
    __test()

# 对特征图features以基准长度为16、选择合适的ratios和scales取基准锚点
 # anchor_base。（选择长度为16的原因是图片大小为600*800左右，基准长度
 # 16对应的原图区域是256*256，考虑放缩后的大小有128*128，512*512比较合适）
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    # 根据基准点生成9个基本的anchor的功能，ratios=[0.5,1,2],anchor_scales=
    # [8,16,32]是长宽比和缩放比例,anchor_scales也就是在base_size的基础上再增
    # 加的量，本代码中对应着三种面积的大小(16*8)^2 ,(16*16)^2  (16*32)^2  
    # 也就是128,256,512的平方大小

    py = base_size / 2.
    px = base_size / 2.

    #（9，4），注意：这里只是以特征图的左上角点为基准产生的9个anchor,
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    # six.moves 是用来处理那些在python2 和 3里面函数的位置有变化的，
    # 直接用six.moves就可以屏蔽掉这些变化
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            # 生成9种不同比例的h和w
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            # 计算出anchor_base画的9个框的左下角和右上角的4个anchor坐标值
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
