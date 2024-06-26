import torch
import torch.nn as nn
from models.ConvTran.convtran import CasualConvTran, CasualConvTran_S
from models.encoder.common import Conv1dEncoder, Conv2dEncoder
import random


class S_1_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.mAcc_encoder = CasualConvTran_S(2653, 512, 3)
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512 * 2, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, mAcc , mLight, wHr, wLight) :
        embed_ma = self.mAcc_encoder(mAcc)
        embed_wh = self.wHr_encoder(wHr)
        embed_vec = torch.cat([embed_ma, embed_wh], dim  = 1)
        x = self.cf(embed_vec)
        x = self.out(x)
        x = self.sig(x)
        return x

class S_2_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512 * 2, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, mAcc , mLight, wHr, wLight) :
        embed_wh = self.wHr_encoder(wHr)
        x = self.cf(embed_wh)
        x = self.out(x)
        x = self.sig(x)
        return x


class S_3_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.mAcc_encoder = CasualConvTran_S(2653, 512, 3)
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512 * 2, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, mAcc , mLight, wHr, wLight) :
        embed_ma = self.mAcc_encoder(mAcc)
        embed_wh = self.wHr_encoder(wHr)
        embed_vec = torch.cat([embed_ma, embed_wh], dim  = 1)
        x = self.cf(embed_vec)
        x = self.out(x)
        x = self.sig(x)
        return x
        
class S_4_Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.mAcc_encoder = CasualConvTran_S(2653, 512, 3)
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.cf = nn.Sequential(nn.Linear(512 * 2, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, mAcc , mLight, wHr, wLight) :
        embed_ma = self.mAcc_encoder(mAcc)
        embed_wh = self.wHr_encoder(wHr)
        embed_vec = torch.cat([embed_ma, embed_wh], dim  = 1)
        x = self.cf(embed_vec)
        x = self.out(x)
        x = self.sig(x)
        return x