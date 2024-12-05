import torch
import torch.nn as nn
import torch.nn.functional as F

from LModel.SCI.model import Finetunemodel


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Finetunemodel('./LModel/SCI/medium.pt')

    def forward(self, x):
        return self.model(x)[1]
