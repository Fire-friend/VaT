import time

from publicutil.comutils import getPackByNameUtil, freeze, show_tensor
import torch


class Trainer:
    def __init__(self, args):
        self.args = args
        # 定义low level模型--------------
        LModel = getPackByNameUtil(py_name='LModel.' + args.LModel + '.network', object_name='Model')
        self.LModel = LModel().cuda()
        freeze(self.LModel, eval=True)

        # # 定义high level模型--------------
        # HModel = getPackByNameUtil(py_name='HModel.' + args.HModel + '.network', object_name='Yolov3')
        # self.HModel = HModel().cuda()
        # freeze(self.HModel, eval=True)

        # 定义Translator trainer--------------
        TModel_trainer = getPackByNameUtil(py_name='VaTrainer.' + args.TTrainer, object_name='Trainer')
        self.TModel_trainer = TModel_trainer(args)

        # self.visualizer = Visualizer(args)

    def set_input(self, input):
        self.input_L = input['A'].cuda()
        self.input_H = input['B'].cuda()
        self.image_paths = input['A_paths']
        self.mask_A = input['A_cutout_mask'].cuda()
        self.mask_B = input['B_cutout_mask'].cuda()

    def train(self):
        # st = time.time()
        with torch.no_grad():
            # output_L = self.LModel((self.input_L-0.5) / 0.5).clamp(0, 1)
            output_L = self.LModel(self.input_L).clamp(0, 1)
        # show_tensor(output_L.permute([0, 2, 3, 1])[1])
        # show_tensor(self.input_L.permute([0, 2, 3, 1])[1])
        self.TModel_trainer.iter(output_L.detach(), self.input_H, self.input_L, self.mask_A, self.mask_B)
        # print(time.time() - st)
