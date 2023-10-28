# written by Shihao Shao (shaoshihao@pku.edu.cn)
from torch import nn

class Relup(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, alpha=0.014):
        super(Relup, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        x = x.clamp(self.alpha)
        return x