import torch
import torch.nn as nn
from models.layers.transformer_encoder_layer import Transformer_encoder


class Conv2dEncoder(nn.Module) :
    def __init__(self, encoded_len, conv_window_size) :
        super().__init__()
        self.conv2dBlock = nn.Sequential(nn.Conv2d(1, encoded_len * 4, kernel_size=[1,conv_window_size], padding='same'),                                        nn.BatchNorm2d(encoded_len*4),nn.GELU())
        self.conv2dBlock2 = nn.Sequential(nn.Conv2d(encoded_len * 4, encoded_len, kernel_size = [3,1], padding = 'valid'), nn.BatchNorm2d(encoded_len), nn.GELU())
        self.poolingBlock = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x) :
        x = x.unsqueeze(1) # (batch, 1 ,3, seq_len)
        x = self.conv2dBlock(x)  #(1, encoded_len * 4, 3, seq_len)
        x = self.conv2dBlock2(x).squeeze(2)# (1, encoded_len, 1, seq_len) -> (1, encoded_len, seq_len)
        x = self.poolingBlock(x).squeeze(2)
        return x


#input (batch, seq)
class Conv1dEncoder(nn.Module) :
    def __init__(self, encoded_len, kernel_size = 3) :
        super().__init__()
        self.convBlock = nn.Sequential(
        nn.Conv1d(1, out_channels = encoded_len, kernel_size = kernel_size + 5),
        nn.BatchNorm1d(encoded_len),
        nn.GELU(),
        nn.Conv1d(encoded_len, out_channels = encoded_len, kernel_size = kernel_size + 2),
        nn.BatchNorm1d(encoded_len),
        nn.GELU(),
        nn.Conv1d(encoded_len, encoded_len, kernel_size = kernel_size),
        nn.BatchNorm1d(encoded_len),
        nn.GELU())

        self.poolingBlock = nn.AdaptiveAvgPool2d(output_size = [1, encoded_len])
    
    def forward(self, x) :
        x = x.unsqueeze(1)
        x = self.convBlock(x)
        x = self.poolingBlock(x).squeeze(1)
        return x
        

        