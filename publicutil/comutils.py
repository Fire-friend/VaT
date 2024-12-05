import importlib

import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

# from HModel.YOLOV3.utils import tools
# from HModel.YOLOV3.utils.tools import xywh2xyxy
import torch

from HModel.YOLOV5.utils.general import xywh2xyxy


def freeze(model, eval=False):
    for name, value in model.named_parameters():
        value.requires_grad = False
    if eval:
        model.eval()


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    data = yaml.load(file_data, yaml.FullLoader)
    file.close()
    return data


def set_yaml_to_args(args, dict: dict):
    for key, val in dict.items():
        args.__setattr__(key, val)

    head_save = './checkSave/' + str(args.model)

    # args.log_path = head_save + '/' + 'log.txt'
    args.log_path = head_save + '/log/'
    args.save_path_img = head_save + '/temp_results/'
    args.save_path_model = head_save + '/checkpoint/'

    if not os.path.exists(args.save_path_img):
        os.makedirs(args.save_path_img)
    if not os.path.exists(args.save_path_model):
        os.makedirs(args.save_path_model)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


def getPackByNameUtil(py_name, object_name):
    module_object = importlib.import_module(py_name)
    object = getattr(module_object, object_name)
    return object


def show_tensor(tensor, mode='RGB'):
    im = tensor.detach().cpu().numpy()
    if mode == 'RGB':
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=mode)
    plt.savefig('temp.png')
    plt.show()


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def convert_pred_torch(pred_bbox, test_input_size, org_img_shape, valid_scale, return_mask=False, th=0.01):
    """
    预测框进行过滤，去除尺度不合理的框
    """
    pred_coor = xywh2xyxy(pred_bbox[:, :4])
    pred_conf = torch.sigmoid(pred_bbox[:, 4])
    pred_prob = torch.sigmoid(pred_bbox[:, 5:])

    # (1)
    # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
    # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
    # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
    # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (2)将预测的bbox中超出原图的部分裁掉
    pred_coor = torch.cat([torch.maximum(pred_coor[:, :2], torch.tensor([0, 0]).cuda()),
                           torch.minimum(pred_coor[:, 2:], torch.tensor([org_w - 1, org_h - 1]).cuda())], dim=-1)
    # (3)将无效bbox的coor置为0
    invalid_mask = torch.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4)去掉不在有效范围内的bbox
    temp = pred_coor[:, 2:4] - pred_coor[:, 0:2]
    bboxes_scale = torch.sqrt(temp[:, 0] * temp[:, 1])
    scale_mask = torch.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5)将score低于score_threshold的bbox去掉
    classes = torch.argmax(pred_prob, dim=-1)
    scores = pred_conf * pred_prob[torch.arange(len(pred_coor)), classes]
    # scores = pred_conf
    score_mask = scores > th

    mask = torch.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]

    bboxes = torch.cat([coors, scores[:, np.newaxis], classes[:, np.newaxis]], dim=-1)
    if return_mask:
        return bboxes, mask
    else:
        return bboxes
