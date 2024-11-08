import torch
import torch.nn as nn
from torch.nn import init
from .residual_attention_module import ResidualAttentionModule
from .multi_head_attention import MultiHeadAttention

from .DWT_IDWT_layer import DWT_1D

class Downsample(nn.Module):
    def __init__(self, args, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.args = args
        self.dwt = DWT_1D(args=self.args, wavename = wavename)

    def forward(self, input):
        LL, HH = self.dwt(input)
        return LL



class ResidualAttention(nn.Module):
    '''
    单尺度残差注意力机制
    '''
    def __init__(self,args, in_channel, out_channels, d_v, d_v_res, n_head=3, dropout=0.1, bias=False):
        super(ResidualAttention, self).__init__()
        self.args = args

        self.layer01_res_attn = nn.Sequential(
            ResidualAttentionModule(args=self.args, in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=1,
                                    n_head=n_head, d_model=out_channels, d_k=64, d_v=d_v_res, dropout=dropout,
                                    padding=1, bias=bias)
        )

        self.layer02_res_attn = nn.Sequential(
            ResidualAttentionModule(args=self.args, in_channels=in_channel, out_channels=out_channels, kernel_size=7, stride=1,
                                    n_head=n_head, d_model=out_channels, d_k=64, d_v=d_v_res, dropout=dropout,
                                    padding=3, bias=bias)
        )

        self.layer03_res_attn = nn.Sequential(
            ResidualAttentionModule(args=self.args, in_channels=in_channel, out_channels=out_channels, kernel_size=17, stride=1,
                                    n_head=n_head, d_model=out_channels, d_k=64, d_v=d_v_res, dropout=dropout,
                                    padding=8, bias=bias)
        )

        self.layer04_attention = MultiHeadAttention(n_head=n_head, d_model=out_channels, d_k=64, d_v=d_v)

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
        x1, attn1 = self.layer01_res_attn(x)
        x2, attn2 = self.layer02_res_attn(x)
        x3, attn3 = self.layer03_res_attn(x)

        ## 注意力机制
        ### 位置编码
        (x1_channel, x1_channel_index), (x2_channel, x2_channel_index), (x3_channel, x3_channel_index) = torch.max(x1, dim=2), \
                                                                                                         torch.max(x2, dim=2), \
                                                                                                         torch.max(x3, dim=2)

        ### 得到key
        x1_channel, x2_channel, x3_channel = torch.squeeze(x1_channel),\
                                             torch.squeeze(x2_channel), \
                                             torch.squeeze(x3_channel)
        x1_channel, x2_channel, x3_channel = torch.unsqueeze(x1_channel, dim=1), \
                                             torch.unsqueeze(x2_channel, dim=1), \
                                             torch.unsqueeze(x3_channel, dim=1)
        key = torch.cat([x1_channel, x2_channel, x3_channel], dim=1)

        ### 得到value
        batch_size, channel_size, dim_size = x1.shape
        x1, x2, x3 = torch.unsqueeze(x1, dim=1),\
                     torch.unsqueeze(x2, dim=1),\
                     torch.unsqueeze(x3, dim=1)
        value = torch.cat([x1, x2, x3], dim=1).view(batch_size, 3, -1)

        out, attn4 = self.layer04_attention(value, key, key)
        out = torch.sum(out, dim=1)
        out = out.contiguous().view(batch_size, channel_size, dim_size)

        out = self.layer_pool(out)

        return out, [attn1, attn2, attn3, attn4]
