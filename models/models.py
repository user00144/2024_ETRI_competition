import torch
import torch.nn as nn
from models.ConvTran.convtran import CasualConvTran, CasualConvTran_S, ConvTran
from models.encoder.common import Conv1dEncoder, Conv2dEncoder
import random

class Q_Model(nn.Module) :
    def __init__(self) :
#        self.conv2d_mAcc = Conv2dEncoder(encoded_len, conv_window_size = 8)
        super().__init__()
        self.mLight_encoder = CasualConvTran(18, 512, 1)#Conv1dEncoder(256, kernel_size = 3) 
        self.wHr_encoder = CasualConvTran(1240//8, 512, 1) #Conv1dEncoder(256, kernel_size = 3) 
        self.wLight_encoder = CasualConvTran(128//8, 512, 1) ##Conv1dEncoder(256, kernel_size = 2)
        #self.cf = nn.Sequential(nn.Linear(256, 128) , nn.ReLU(), nn.Dropout(0.3))
        #self.cf = nn.Sequential(nn.Linear(512 * 3, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 256),nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128))
        #self.cf = nn.Sequential(nn.Linear(512 * 3, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        #self.out = nn.Linear(128, 1)
        self.out = ConvTran(1, 3, 64)
        self.sig = nn.Sigmoid()
  #      self.alpha = 0.0
    # mAcc ( batch , 3 , seq_len)  / mLight , wHr, w Light ( batch, seq_len )
    def forward(self, mLight, wHr, wLight) :
        embed_ml = self.mLight_encoder(mLight)
        embed_wh = self.wHr_encoder(wHr)
        embed_wl = self.wLight_encoder(wLight)

        embed_vec = torch.stack([embed_ml ,embed_wl, embed_wh], dim  = 1)
        #x = self.cf(embed_vec)
        x = self.out(embed_vec)
        x = self.sig(x)
        #if random.random() < self.alpha :
        #    return self.one - x
        #else :
        #    return x
        return x


class S_Model(nn.Module) :
    def __init__(self) :
#        self.conv2d_mAcc = Conv2dEncoder(encoded_len, conv_window_size = 8)
        super().__init__()
        self.mLight_encoder = CasualConvTran(9, 512, 1)#Conv1dEncoder(256, kernel_size = 3) 
        self.wHr_encoder = CasualConvTran(78, 512, 1) #Conv1dEncoder(512, kernel_size=5) #
        self.wLight_encoder = CasualConvTran(128//16, 512, 1) #Conv1dEncoder(512, kernel_size=2) 
       #self.cf = nn.Sequential(nn.Linear(256, 128) , nn.ReLU(), nn.Dropout(0.3))
        #self.cf = nn.Sequential(nn.Linear(512 * 2, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        #self.alpha = 0.0
        self.out = ConvTran(1, 3, 64)
        self.sig = nn.Sigmoid()
    # mAcc ( batch , 3 , seq_len)  / mLight , wHr, w Light ( batch, seq_len )
    def forward(self, mLight, wHr, wLight) :
        embed_wh = self.wHr_encoder(wHr)
        embed_wl = self.wLight_encoder(wLight)
        embed_ml = self.mLight_encoder(mLight)

        embed_vec = torch.stack([embed_wh, embed_ml, embed_wl], dim  = 1)
        #x = self.cf(embed_vec)
        x = self.out(embed_vec)
        x = self.sig(x)
        #if random.random() < self.alpha :
        #    return self.one - x
        #else :
        #    return x
        return x



class Q_Acc_Model(nn.Module) :
    def __init__(self) :
#        self.conv2d_mAcc = Conv2dEncoder(encoded_len, conv_window_size = 8)
        super().__init__()
        self.mAcc_encoder = CasualConvTran_S(5305, 512, 3)
        self.mLight_encoder = CasualConvTran(18, 512, 1)#Conv1dEncoder(256, kernel_size = 3) 
        self.wHr_encoder = CasualConvTran(1240//8, 512, 1) #Conv1dEncoder(256, kernel_size = 3) 
        #self.cf = nn.Sequential(nn.Linear(256, 128) , nn.ReLU(), nn.Dropout(0.3))
        #self.cf = nn.Sequential(nn.Linear(512 * 3, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 256),nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128),nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64))
        #self.cf = nn.Sequential(nn.Linear(512 * 3, 512),nn.ReLU(), nn.Dropout(0.3) ,nn.Linear(512, 128),nn.ReLU(), nn.Dropout(0.3))
        self.out = ConvTran(1, 3, 64)
        self.sig = nn.Sigmoid()
        #self.alpha = 0.0
    # mAcc ( batch , 3 , seq_len)  / mLight , wHr, w Light ( batch, seq_len )
    def forward(self, mAcc ,mLight, wHr, wLight) :
        embed_ma = self.mAcc_encoder(mAcc)
        embed_ml = self.mLight_encoder(mLight)
        embed_wh = self.wHr_encoder(wHr)

        embed_vec = torch.stack([embed_ma, embed_ml, embed_wh], dim  = 1)
        #x = self.cf(embed_vec)
        x = self.out(embed_vec)
        x = self.sig(x)
        #if random.random() < self.alpha :
        #    return self.one - x
        #else :
        #    return x
        return x


    