import torch
import torch.nn as nn
from models.ConvTran.convtran import CasualConvTran, CasualConvTran_S
from models.encoder.common import Conv1dEncoder, Conv2dEncoder
import random


class S_1_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512, 256),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(256, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self , mLight, wHr, wLight) :
        embed_wh = self.wHr_encoder(wHr)
        x = self.cf(embed_wh)
        x = self.out(x)
        x = self.sig(x)
        return x

class S_2_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512, 256),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(256, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self , mLight, wHr, wLight) :
        embed_wh = self.wHr_encoder(wHr)
        x = self.cf(embed_wh)
        x = self.out(x)
        x = self.sig(x)
        return x


class S_3_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512, 256),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(256, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self , mLight, wHr, wLight) :
        embed_wh = self.wHr_encoder(wHr)
        x = self.cf(embed_wh)
        x = self.out(x)
        x = self.sig(x)
        return x

class S_4_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512, 256),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(256, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self , mLight, wHr, wLight) :
        embed_wh = self.wHr_encoder(wHr)
        x = self.cf(embed_wh)
        x = self.out(x)
        x = self.sig(x)
        return x
