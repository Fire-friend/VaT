# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets, th_cer1=0.5, th_cer2=0.2, th_un=0.99, th_cls=0.95,
                 th_obj=0.65):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        # 可靠的样本索引
        certain_index = (targets[:, 6] > th_cer1) * (targets[:, 7] < th_un)
        uncertain_index = (targets[:, 6] > th_cer2) * (targets[:, 6] < th_cer1) * (targets[:, 7] < th_un)
        cls_certain_index = uncertain_index * (targets[:, 9] > th_cls)
        iou_certain_index = uncertain_index * (targets[:, 8] > th_obj)

        tcls, tbox, indices, anchors, als, uns, confs, obj_confs, \
            cls_confs = self.build_targets(p, targets[certain_index])
        _, _, un_indices, _, un_als, _, un_confs, _, _ = self.build_targets(p, targets[uncertain_index])
        un_tcls, _, un_cls_indices, _, un_cls_als, _, _, _, _ = self.build_targets(p, targets[cls_certain_index])
        _, un_tbox, un_obj_indices, un_anchors, un_obj_als, _, _, _, _ = self.build_targets(p, targets[iou_certain_index])

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            # print(torch.sum(roi_index))
            alpha_tobj = torch.zeros_like(tobj) + torch.mean(als[i])
            n = b.shape[0]  # number of targets
            if n:
                # certain loss ----------------------
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                iou_loss = (als[i] * (1.0 - iou)).mean()  # iou loss
                if torch.isnan(iou_loss):
                    iou_loss = 0
                lbox += iou_loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou
                alpha_tobj[b, a, gj, gi] = als[i].type(iou.dtype)

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    cls_loss = torch.mean(als[i].unsqueeze(1) * self.BCEcls(pcls, t))  # BCE
                    if torch.isnan(cls_loss):
                        cls_loss = 0
                    lcls += cls_loss
                # -----------------------------------------------
                # uncertain label cal obj loss
                uc_b, uc_a, uc_gj, uc_gi = un_indices[i]
                n = uc_b.shape[0]
                if n:
                    tobj[uc_b, uc_a, uc_gj, uc_gi] = un_confs[i].type(tobj.dtype)
                    alpha_tobj[uc_b, uc_a, uc_gj, uc_gi] = un_als[i].type(iou.dtype)

                # uncertain label cal Regression
                uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi = un_obj_indices[i]
                n = uc_obj_b.shape[0]
                uc_tbox_spec = un_tbox[i]
                anchors_spec = un_anchors[i]
                if n:
                    uc_ps = pi[uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi]
                    pxy = uc_ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (uc_ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, uc_tbox_spec, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    iou_loss = (un_obj_als * (1.0 - iou)).mean()  # iou loss
                    if torch.isnan(iou_loss):
                        iou_loss = 0
                    lbox += iou_loss

                # uncertain label cal Classification
                # uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi = un_cls_indices[i]
                # n = uc_cls_b.shape[0]
                # if n:
                #     uc_ps = pi[uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi]
                #     if self.nc > 1:  # cls loss (only if multiple classes)
                #         t = torch.full_like(uc_ps[:, 5:], self.cn, device=self.device)  # targets
                #         t[range(n), un_tcls[i]] = self.cp
                #         cls_loss = torch.mean(
                #             un_cls_als[i].unsqueeze(1) * self.BCEcls(uc_ps[:, 5:], t))
                #         if torch.isnan(cls_loss):
                #             cls_loss = 0
                #         lcls += cls_loss

            valid_mask = tobj >= 0
            # obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
            obji = torch.mean(self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask]) * alpha_tobj[valid_mask])
            if torch.isnan(obji):
                obji = 0
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, als, uns, confs, obj_confs, cls_confs = [], [], [], [], [], [], [], [], []
        gain = torch.ones(12, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        temp = targets.repeat(na, 1, 1)
        targets = torch.cat((temp[:, :, :-5], ai[..., None], temp[:, :, -5:]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
                torch.tensor(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                        [-1, 0],
                        [0, -1],
                    ],
                    device=self.device,
                ).float()
                * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            conf = t[:, 7]
            un = t[:, 8]
            obj_conf = t[:, 9]
            cls_conf = t[:, 10]
            al = t[:, 11]
            bc, gxy, gwh, a = t[:, :7].chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            als.append(al)
            uns.append(un)
            confs.append(conf)
            obj_confs.append(obj_conf)
            cls_confs.append(cls_conf)

        return tcls, tbox, indices, anch, als, uns, confs, obj_confs, cls_confs
