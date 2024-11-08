import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from torch.nn import init
from .DWT_IDWT_layer import DWT_1D

class Downsample(nn.Module):
    def __init__(self,args, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.args = args
        self.dwt = DWT_1D(args, wavename = wavename)

    def forward(self, input):
        LL, HH = self.dwt(input)
        return LL

class ResidualAttentionModule(nn.Module):
    '''
    残差注意力模块
    '''
    def __init__(self,args, in_channels, out_channels, kernel_size, stride,
                 n_head, d_model, d_k, d_v, dropout=0.1, padding=0, bias=False):
        super(ResidualAttentionModule, self).__init__()
        self.args = args

        self.layer01 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.SELU(inplace=True)
        )

        self.layer02 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.SELU(inplace=True),
            # nn.MaxPool1d(2)
            Downsample(args=self.args)
        )

        self.layer_keep_channel = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1,), stride=(1,), padding=0,
                      bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.SELU(inplace=True),
            # nn.MaxPool1d(2)
            Downsample(args=self.args)
        )

        self.layer_attention = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)

        self.layer_pool = nn.Sequential(
            nn.SELU(inplace=True),
            # nn.MaxPool1d(2)
            Downsample(args=self.args)
        )

        ## kaiming初始化
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        ## 卷积提取特征
        layer01_x = self.layer01(x)
        layer02_x = self.layer02(layer01_x)
        ## 升x的通道
        x = self.layer_keep_channel(x)
        ## 注意力机制
        ### 位置编码
        (x_channel, x_channel_index), (layer02_x_channel, layer02_x_channel) = torch.max(x, dim=2), torch.max(layer02_x, dim=2)
        ### 得到key
        x_channel, layer02_x_channel = torch.squeeze(x_channel), torch.squeeze(layer02_x_channel)
        x_channel, layer02_x_channel = torch.unsqueeze(x_channel, dim=1), torch.unsqueeze(layer02_x_channel, dim=1)
        key = torch.cat([x_channel, layer02_x_channel], dim=1)

        ### 得到value
        batch_size, channel_size, dim_size = x.shape
        x, layer02_x = torch.unsqueeze(x, dim=1), torch.unsqueeze(layer02_x, dim=1)
        value = torch.cat([x, layer02_x], dim=1).view(batch_size, 2, -1)

        out, attn = self.layer_attention(value, key, key)
        out = torch.sum(out, dim=1)
        out = out.contiguous().view(batch_size, channel_size, dim_size)

        out = self.layer_pool(out)
        return out, attn