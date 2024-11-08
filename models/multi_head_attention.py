import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    '''
    Multi Head Attention
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        '''

        :param n_head: 多头的数量
        :param d_model: embedding的维度
        :param d_k: Key的维度
        :param d_v: Value的维度
        :param dropout:
        '''
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False) # 改动输入的为原始值的维度，不是embedding的维度
        # self.fc = nn.Linear(n_head * d_v, d_v, bias=False) # 改动输入的为原始值的维度，不是embedding的维度

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_v, eps=1e-6)

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

    def forward(self, v, q, k, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        ## size_batch: batch的大小
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = v
        ## embedding * 矩阵得到q，k， v
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        ## 变换维度
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        ## 最后的
        attn = self.attention(q, k, v, mask=mask)

        # q = q.transpose(1,2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        # q += residual

        attn = torch.sum(attn, dim=1) / attn.shape[1]
        attn = torch.squeeze(attn)

        out = torch.matmul(attn, v)

        out = self.layer_norm(out)
        return out, attn
