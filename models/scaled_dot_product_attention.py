import torch
import torch.nn as nn
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

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

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q/self.temperature, k.transpose(2,3))

        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)

        attn = self.dropout(torch.softmax(attn, dim=-1))
        ### 将value仅仅加一个
        # output = torch.matmul(attn, v)
        return attn