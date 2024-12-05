import argparse
import csv
import os
import platform
import random
import sys
from pathlib import Path

import kornia
import matplotlib.pyplot as plt
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = './HModel/YOLOV5'  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh, xyxy2xywhn,
)
from utils.torch_utils import select_device, smart_inference_mode


def yolov5_mosaic(ori_img: torch.Tensor, ori_label, images: torch.Tensor, labels: list):
    C, H, W = ori_img.shape
    mosaic_img = torch.full((C, H, W), 0.5, dtype=images.dtype)
    mosaic_labels = []

    yc, xc = random.randint(H // 2, H), random.randint(W // 2, W)
    indices = random.sample(range(len(images)), 4)
    images[indices[0]] = ori_img
    labels[indices[0]] = ori_label
    for i, idx in enumerate(indices):
        img = images[idx]
        lbl = labels[idx]

        if i == 0:  # Top-left
            x1, y1, x2, y2 = max(xc - W, 0), max(yc - H, 0), xc, yc
            img_x1, img_y1, img_x2, img_y2 = W - (x2 - x1), H - (y2 - y1), W, H
        elif i == 1:  # Top-right
            x1, y1, x2, y2 = xc, max(yc - H, 0), min(xc + W, W), yc
            img_x1, img_y1, img_x2, img_y2 = 0, H - (y2 - y1), x2 - x1, H
        elif i == 2:  # Bottom-left
            x1, y1, x2, y2 = max(xc - W, 0), yc, xc, min(yc + H, H)
            img_x1, img_y1, img_x2, img_y2 = W - (x2 - x1), 0, W, y2 - y1
        elif i == 3:  # Bottom-right
            x1, y1, x2, y2 = xc, yc, min(xc + W, W), min(yc + H, H)
            img_x1, img_y1, img_x2, img_y2 = 0, 0, x2 - x1, y2 - y1

        mosaic_img[:, y1:y2, x1:x2] = img[:, img_y1:img_y2, img_x1:img_x2]

        if lbl is not None and len(lbl) > 0:
            for box in lbl:
                orig_x_min, orig_y_min, orig_x_max, orig_y_max = box[:4]
                x_min = orig_x_min + x1 - img_x1
                y_min = orig_y_min + y1 - img_y1
                x_max = orig_x_max + x1 - img_x1
                y_max = orig_y_max + y1 - img_y1

                if x_min < W and x_max > 0 and y_min < H and y_max > 0:
                    if min(x_max, W) - max(x_min, 0) > 6 and min(y_max, H) - max(y_min, 0) > 6:
                        mosaic_labels.append(
                            [
                                max(x_min, 0),
                                max(y_min, 0),
                                min(x_max, W),
                                min(y_max, H),
                                *box[4:].tolist(),
                            ]
                        )

    mosaic_labels = torch.tensor(mosaic_labels, dtype=torch.float32)
    return mosaic_img, mosaic_labels

def random_data_augmentation(images: torch.Tensor, labels: list):
    """
    Perform random data augmentation on input images and labels for object detection.

    Args:
        images (torch.Tensor): Input images with shape (N, C, H, W).
        labels (list): A list of length N, where each element is a tensor of shape (K, 8),
                       representing [x1, y1, x2, y2, class, overall confidence, regression confidence, classification confidence].

    Returns:
        torch.Tensor, list: Augmented images and labels.
    """
    N, C, H, W = images.shape

    # Random horizontal flip
    if random.random() < 0.5:
        images = kornia.geometry.transform.hflip(images)
        # Update labels
        for i in range(N):
            if labels[i] is not None and len(labels[i]) > 0:
                boxes = labels[i][:, :4]  # Extract [x1, y1, x2, y2]
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]  # Flip x coordinates
                labels[i][:, :4] = boxes

    # Random brightness adjustment
    if random.random() < 0.5:
        gamma = random.uniform(0.5, 1.3)  # Gamma range: <1 darken, >1 brighten
        gain = random.uniform(0.8, 1.1)  # Gain factor for overall brightness
        images = kornia.enhance.adjust_gamma(images, gamma, gain)

    # Random scaling down
    if random.random() < 0.5:
        scale_factor = random.uniform(0.5, 1.0)
        new_h, new_w = int(H * scale_factor), int(W * scale_factor)

        # Scale down the image
        resized_images = kornia.geometry.transform.resize(images, (new_h, new_w))
        images = torch.zeros_like(images)  # Create a blank image container
        pad_h = (H - new_h) // 2
        pad_w = (W - new_w) // 2
        images[:, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_images

        # Update labels
        for i in range(N):
            if labels[i] is not None and len(labels[i]) > 0:
                boxes = labels[i][:, :4]
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_factor + pad_w
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_factor + pad_h
                labels[i][:, :4] = boxes

    # Random Cutout
    images_s = images.clone()
    labels_s = labels.copy()
    for i in range(N):
        if random.random() < 0.8:
            # Random Mosaic
            mosaic_out = yolov5_mosaic(images[i], labels[i], images_s, labels_s)
            images[i] = mosaic_out[0]
            labels[i] = mosaic_out[1]
        if labels[i] is not None and len(labels[i]) > 0:
            for box in labels[i]:
                if random.random() < 0.5:
                    x1, y1, x2, y2 = box[:4].int()
                    cutout_h = random.randint(1, (y2 - y1) // 2)
                    cutout_w = random.randint(1, (x2 - x1) // 2)
                    cutout_x = random.randint(x1, x2 - cutout_w)
                    cutout_y = random.randint(y1, y2 - cutout_h)
                    images[i, :, cutout_y:cutout_y + cutout_h, cutout_x:cutout_x + cutout_w] = 0
    return images, labels


class GetPseudo():
    def __init__(self,
                 weights=ROOT + "/best.pt",  # model path or triton URL
                 data=ROOT + "/voch.yaml",  # dataset.yaml path
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.001,  # confidence threshold
                 iou_thres=0.6,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device="cuda:0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 ):
        # 加载模型
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        for param in self.model.parameters():
            param.requires_grad = False
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

    def getPseudo(self, im, im2=None, show=False, nuq=None, aug=False):
        """
        Args:
            im:
            show:

        Returns: [index, class, x1, y1, x2, y2, confidence]

        """
        source_im = im.clone()
        n, c, h, w = source_im.shape
        # Inference
        pred = self.model(im, augment=False, visualize=False)
        if isinstance(pred, (list, tuple)):
            pred_f = pred[1]
            pred = pred[0]

        if im2 is not None:
            for im_c in im2:
                pred_c = self.model(im_c, augment=False, visualize=False)
                if isinstance(pred_c, (list, tuple)):
                    pred_c_f = pred_c[1]
                    pred_c = pred_c[0]
                pred = torch.cat([pred, pred_c], dim=1)
                pred_f = torch.cat([pred_f, pred_c_f], dim=1)

        # NMS
        pred, pred_ori, pred_f = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                                     self.classes, self.agnostic_nms, max_det=self.max_det,
                                                     return_ori=True,
                                                     preds_f=pred_f)
        for i in range(len(pred)):
            cls_pro = torch.max(pred_ori[i][:, 5:], dim=1, keepdim=True)[0]
            # NUQ uncertainty
            if pred_f[i] is not None:
                nuq_out = nuq.predict(pred_f[i])
                nuq_prob, un_alea, un_epis = nuq_out[:, 0], torch.exp(nuq_out[:, 1]), torch.exp(nuq_out[:, 2])
            else:
                un_epis = torch.zeros_like(cls_pro).squeeze(1)
            pred[i] = torch.cat([pred[i], pred_ori[i][:, 4:5], cls_pro, un_epis.unsqueeze(1)], dim=1)
        names = self.model.names
        if aug:
            im, pred = random_data_augmentation(images=im, labels=pred)

        # Process predictions
        result = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                cur_det = torch.zeros(size=(len(det), 10)).cuda() + i
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], (h, w, c)).round()
                if show:
                    annotator = Annotator(im.permute([0, 2, 3, 1]).contiguous()[i].detach().cpu().numpy(),
                                          line_width=3, example=str(names))
                    for *xyxy, conf, cls, obj_conf, cls_conf, un in reversed(det):
                        c = int(cls)  # integer class
                        # iou = nuq_prob[- jj - 1]
                        # if un > 0.7 and iou < 0.4:
                        #     continue
                        label = f"{names[c]} {conf:.2f} {un:.2f}"
                        annotator.box_label(xyxy, label, color=(0, 0, 255))

                det[:, :4] = xyxy2xywhn(det[:, :4], w=w, h=h, clip=True, eps=1e-3)
                cur_det[:, 2:6] = det[:, :4]
                cur_det[:, 1] = det[:, 5]
                cur_det[:, 6] = det[:, 4]
                cur_det[:, 7] = det[:, 8]# + un_alea
                cur_det[:, 8] = det[:, 6]
                cur_det[:, 9] = det[:, 7]
                result.append(cur_det)
                if show:
                    im0 = annotator.result()
                    plt.imshow(im0)
                    plt.savefig('./temp.png')

        if len(result) > 0:
            out = torch.cat(result, dim=0)
            out[:, 7] = out[:, 7] * 500 + 0.5
            return im, out
        else:
            return None

if __name__ == '__main__':
    pass
