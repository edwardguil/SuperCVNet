import torch
from torch import nn
import torch.functional as F
    
# written by Shihao Shao (shaoshihao@pku.edu.cn)
class Sgem(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, ps=10., infinity = True):
        super(Sgem, self).__init__()
        self.ps = ps
        self.infinity = infinity
    def forward(self, x):

        x = torch.stack(x,0)

        if self.infinity:
            x = F.normalize(x, p=2, dim=-1) # 3 C
            x = torch.max(x, 0)[0] 
        else:
            gamma = x.min()
            x = (x - gamma).pow(self.ps).mean(0).pow(1./self.ps) + gamma

        return x