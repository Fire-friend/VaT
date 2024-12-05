import os
import pickle
import time
from collections import OrderedDict
import cv2
import kornia
import numpy as np
import torch
import itertools
from HModel.YOLOV5.getPseudo import GetPseudo
from HModel.YOLOV5.utils.loss import ComputeLoss
from VaTrainer.network import Model
import torch.nn.functional as F
from publicutil import util
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from nuq import NuqClassifier, NuqRegressor
import copy
from publicutil.comutils import freeze, show_tensor
from publicutil.util import plot_grad_flow

class EMA_Teacher():
    def __init__(self, model, alpha=0.99):
        self.ema_model = copy.deepcopy(model)
        freeze(self.ema_model)
        self.alpha = alpha
        self.global_step = 0

    def update(self, model):
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        self.global_step += 1

class Trainer():
    def __init__(self, args):
        self.opt = args
        self.TModel_A = Model(args, input_channel=6).cuda()
        self.TModel_B = Model(args).cuda()
        self.optimize = torch.optim.AdamW(itertools.chain(self.TModel_A.parameters(), self.TModel_B.parameters()),
                                          lr=args.lr, weight_decay=1e-4, betas=(args.beta1, 0.999))
        self.stepLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimize, T_0=1000 * 2, T_mult=2,
                                                                           eta_min=1e-6)
        self.scaler = GradScaler()
        self.getPseudo = GetPseudo(conf_thres=0.1)
        self.compute_loss = ComputeLoss(self.getPseudo.model.model)
        # NUQ------------
        # Original NUQ for tuning bandwith, "from nuq_s import NuqClassifier, NuqRegressor"
        # reload a file to a variable
        with open('NUQ_train_data.pickle', 'rb') as file:
            NUQ_data = pickle.load(file)

        x_NUQ = NUQ_data['x']
        y_NUQ = NUQ_data['y']
        # cls_NUQ = NUQ_data['cls']
        # x_NUQ = torch.from_numpy(x_NUQ).cuda()
        # y_NUQ = torch.from_numpy(y_NUQ).cuda()
        # self.nuq_cls = NuqClassifier(
        #     n_neighbors=10,
        #     # verbose=True,
        # )
        # self.nuq_cls.fit(x_NUQ, cls_NUQ)
        self.nuq_regress = NuqRegressor(
            n_neighbors=20,
            verbose=False,
        )
        self.nuq_regress.fit(x_NUQ, y_NUQ)
        self.H_loss_sum = 0
        self.step = 0

    def iter(self, output_L, input_H, input_L, mask_A, mask_B):
        self.TModel_A.train()
        self.TModel_B.train()
        self.output_L = output_L.clone()  # * mask_A
        self.input_H = input_H.clone()  # * mask_B
        self.input_L = input_L.clone()
        self.mask_A = mask_A
        self.mask_B = mask_B

        with autocast():
            # 1. cycle consistency loss
            self.pseudo_input_H, self.input_f, self.weight_t = self.TModel_A(
                torch.cat([self.output_L, self.input_L], dim=1))
            self.output_L_rec = self.TModel_B(self.pseudo_input_H)
            self.loss_cyc_L = F.l1_loss(self.input_f, self.output_L_rec)  # * 10

            self.pseudo_output_L = self.TModel_B(self.input_H)
            self.input_H_rec = self.TModel_A(self.pseudo_output_L, skip=True)
            self.loss_cyc_H = F.l1_loss(self.input_H, self.input_H_rec)  # * 10

            # self.idt_H = self.TModel_A(self.input_H)
            # self.pseudo_input_H_L = self.TModel_A(self.input_L)
            self.loss_cyc = self.loss_cyc_L + self.loss_cyc_H
            # print('t1:', time.time() - st)
            # st = time.time()

            # 2. MLE loss
            with torch.no_grad():
                self.getPseudo.model.model.eval()
                flag = 0
                # TODO Concatenate high-quality images and low-quality images to allow mosaic to better blend
                #  information from different domains when obtaining strongly augmented images.
                input_H_aug, pseudo_I_h = self.getPseudo.getPseudo(self.input_H.clone(),
                                                                   nuq=self.nuq_regress,
                                                                   aug=False)
                _, pseudo_I_f = self.getPseudo.getPseudo(input_L.clone(),
                                                         im2=[output_L.clone()],
                                                         nuq=self.nuq_regress,
                                                         aug=False)
                input_f_aug, pseudo_I_l = self.getPseudo.getPseudo(self.input_f.clone(),
                                                                   im2=[output_L.clone(),
                                                                        input_L.clone(),
                                                                        self.pseudo_input_H.clone()],
                                                                   nuq=self.nuq_regress, aug=True)
                self.pseudo_add = len(pseudo_I_l) - len(pseudo_I_f)
                if pseudo_I_l is None:
                    flag = 1
                if pseudo_I_h is None:
                    flag = 1
                if pseudo_I_f is None:
                    flag = 1

            if flag == 0:
                self.getPseudo.model.model.train()
                # fusion flow supervision
                alpha = np.random.beta(0.5, 0.5)
                alpha = min(alpha, 1 - alpha)
                mix_input = (1 - alpha) * self.input_f + (alpha) * input_H_aug
                mix_pseudo = torch.cat([pseudo_I_f, pseudo_I_h], dim=0)
                alpha_temp = torch.zeros_like(mix_pseudo[:, 0:1])
                alpha_temp[:len(pseudo_I_f)] = (1 - alpha) * pseudo_I_f[:, 7:8]
                alpha_temp[len(pseudo_I_f):] = alpha * pseudo_I_h[:, 7:8]
                mix_pseudo = torch.cat([mix_pseudo, alpha_temp], dim=1)
                out = self.getPseudo.model.model(mix_input)
                self.MLEloss_mix_f, _ = self.compute_loss(out, mix_pseudo,
                                                          th_cer1=0.5,
                                                          th_cer2=0.2,
                                                          th_un=0.99,
                                                          th_cls=0.99,
                                                          th_obj=0.99)

                # translation flow supervision
                alpha2 = np.random.beta(0.5, 0.5)
                alpha2 = min(alpha2, 1 - alpha2)
                # self.mix_input = (1 - alpha2) * self.pseudo_input_H + (alpha2) * self.input_H
                # cat_input_L = torch.cat([self.output_L, self.input_L], dim=1)
                # self.mix_input = (1 - alpha2) * cat_input_L + (alpha2) * torch.cat([self.input_H, self.input_H], dim=1)
                self.mix_input = (1 - alpha2) * input_f_aug + (alpha2) * input_H_aug
                self.mix_input = self.TModel_A(self.mix_input, skip=True)  #
                self.mix_pseudo = torch.cat([pseudo_I_l, pseudo_I_h], dim=0)
                alpha_temp = torch.zeros_like(self.mix_pseudo[:, 0:1])
                alpha_temp[:len(pseudo_I_l)] = (1 - alpha2) * pseudo_I_l[:, 7:8]
                alpha_temp[len(pseudo_I_l):] = alpha2 * pseudo_I_h[:, 7:8]
                mix_pseudo = torch.cat([self.mix_pseudo, alpha_temp], dim=1)
                out = self.getPseudo.model.model(self.mix_input)
                self.MLEloss_mix = self.compute_loss(out, mix_pseudo,
                                                     th_cer1=0.5,
                                                     th_cer2=0.2,
                                                     th_un=0.99,
                                                     th_cls=0.99,
                                                     th_obj=0.99)[0]

                # high-quality flow supervision
                # with torch.no_grad():
                #     logit_H = self.getPseudo.model.model(self.input_H)
                # out1 = self.getPseudo.model.model(self.pseudo_output_L)
                # self.MLEloss_h = sum(
                #     [F.mse_loss(logit_H[i], out1[i])for i in range(len(out1))]) / len(out1)
                out = self.getPseudo.model.model(self.TModel_A(input_H_aug, skip=True))
                alpha_temp = torch.ones_like(pseudo_I_h[:, 0:1])
                pseudo = torch.cat([pseudo_I_h, alpha_temp], dim=1)
                self.MLEloss_h = torch.sum(
                    self.compute_loss(out, pseudo,
                                      th_cer1=0.5,
                                      th_cer2=0.5,
                                      th_un=0.99,
                                      th_cls=0.99,
                                      th_obj=0.99)[0])

                self.Hloss = self.MLEloss_mix + self.MLEloss_mix_f + self.MLEloss_h
            else:
                self.Hloss = self.loss_cyc
            self.loss = self.Hloss + self.loss_cyc

            self.H_loss_sum += self.Hloss.item()
            self.step += 1

        # plot_grad_flow(self.TModel_A.named_parameters())
        self.optimize.zero_grad()
        self.scaler.scale(self.loss).backward()
        # plot_grad_flow(self.TModel_A.named_parameters())
        self.scaler.step(self.optimize)
        self.scaler.update()
        self.stepLR.step()

    def get_current_errors(self, epoch):
        return OrderedDict(
            [
                ('Cyc_L', self.loss_cyc.item()),
                ('H_L', self.Hloss.item()),
                ('MLEloss_mix', self.MLEloss_mix.item()),
                ('MLEloss_mix_f', self.MLEloss_mix_f.item()),
                ('MLEloss_h', self.MLEloss_h.item()),
                ('H_L_avf', self.H_loss_sum / self.step),
                ('lr^3', self.optimize.state_dict()['param_groups'][0]['lr'] * 1000),
                ('pseudo_add', self.pseudo_add)
            ])

    def get_current_visuals(self):
        output_L = util.tensor2im(self.output_L.data)
        input_mix = util.tensor2im(self.mix_input[:, :3].data)
        pseudo_input_H = util.tensor2im(self.pseudo_input_H.data)
        output_L_rec = util.tensor2im(self.output_L_rec.data)
        input_f = util.tensor2im(self.input_f.data)
        input_L = util.tensor2im(self.input_L.data)
        pseudo_output_L = util.tensor2im(self.pseudo_output_L.data)

        input_H = util.tensor2im(self.input_H.data)
        input_H_rec = util.tensor2im(self.input_H_rec.data)
        weight_t = util.tensor2im(self.weight_t.data)
        weight_t = cv2.applyColorMap(weight_t[:, :, 0], cv2.COLORMAP_JET)
        diff_f_out = util.tensor2im(torch.abs(self.input_f.data - self.pseudo_input_H.data))
        diff_f_out = cv2.applyColorMap(diff_f_out[:, :, 0], cv2.COLORMAP_JET)

        return OrderedDict(
            [('output_L', output_L), ('pseudo_input_H', pseudo_input_H), ('output_L_rec', output_L_rec),
             ('input_H', input_H), ('input_L', input_L), ('input_f', input_f),
             ('input_mix', input_mix), ('input_H_rec', input_H_rec), ('weight', weight_t),
             ('pseudo_output_L', pseudo_output_L), ('diff_out', diff_f_out)
             ])

    def save(self, epoch, args):
        self.TModel_A.eval()
        save_filename = '{}_net_{}.pth'.format(epoch, args.TTrainer)
        save_path = os.path.join('checkSave', args.model, 'checkpoint', save_filename)
        torch.save(self.TModel_A.state_dict(), save_path)
