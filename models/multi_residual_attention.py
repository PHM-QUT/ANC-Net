import torch
import torch.nn as nn
from .residual_attention import ResidualAttention
# from model.multi_head_attention import MultiHeadAttention
from torch.nn import init

class MultiResidualAttention(nn.Module):
    '''
    多尺度残差注意力机制
    '''

    def __init__(self, args):
        super(MultiResidualAttention, self).__init__()
        self.args = args

        self.layer01_multi_res_attn = ResidualAttention(args=self.args, in_channel=1, out_channels=32, d_v=32*512,d_v_res=32*1024, n_head=3)

        self.layer02_multi_res_attn = ResidualAttention(args=self.args, in_channel=32, out_channels=128, d_v=128*64,d_v_res=128*128, n_head=3)

        # self.layer03_multi_res_attn = ResidualAttention(kernel_size=17, stride=1, padding=8, n_head=3)

        # self.layer04_attention = MultiHeadAttention(n_head=3, d_model=128, d_k=64, d_v=128*128)
        #
        # self.layer_pool = nn.Sequential(
        #     nn.Tanh(),
        #     nn.MaxPool1d(2)
        # )

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
        x1, attn1 = self.layer01_multi_res_attn(x)
        x2, attn2 = self.layer02_multi_res_attn(x1)
        # x3, attn3 = self.layer03_multi_res_attn(x)

        ## 注意力机制
        ### 位置编码
        # (x1_channel, x1_channel_index) , (x2_channel, x2_channel_index), (x3_channel, x3_channel_index) = torch.max(x1, dim=2), torch.max(x2, dim=2), torch.max(x3, dim=2)
        #
        # ### 得到key
        # x1_channel, x2_channel, x3_channel = torch.squeeze(x1_channel), torch.squeeze(x2_channel), torch.squeeze(x3_channel)
        # x1_channel, x2_channel, x3_channel = torch.unsqueeze(x1_channel, dim=1), torch.unsqueeze(x2_channel, dim=1), torch.unsqueeze(x3_channel, dim=1)
        # key = torch.cat([x1_channel, x2_channel, x3_channel], dim=1)
        #
        # ### 得到value
        # batch_size, channel_size, dim_size = x1.shape
        # x1, x2, x3 = torch.unsqueeze(x1, dim=1), torch.unsqueeze(x2, dim=1), torch.unsqueeze(x3, dim=1)
        # value = torch.cat([x1, x2, x3], dim=1).view(batch_size, 3, -1)
        #
        # out, attn = self.layer04_attention(value, key, key)
        # out = torch.sum(out, dim=1)
        # out = out.contiguous().view(batch_size, channel_size, dim_size)
        #
        # out = self.layer_pool(out)


        return x2, [attn1, attn2]

