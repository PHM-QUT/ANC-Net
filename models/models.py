import torch

from .multi_residual_attention import MultiResidualAttention
from torch.nn import init
import torch.nn as nn

class Classifier(nn.Module):
    '''
    最终的模型
    '''
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        self.layer01_mul = nn.Sequential(
            MultiResidualAttention(self.args)
        )

        self.layer02_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1)
        )

        self.layer03_fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.SELU(inplace=True)
        )

        self.layer04_fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=10)
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
        x, attn = self.layer01_mul(x)
        x = self.layer02_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer03_fc(x)
        x = self.layer04_fc(x)
        return x, attn