from torch import nn

# written by Shihao Shao (shaoshihao@pku.edu.cn)
class Gemp(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, p=4.6, eps = 1e-8):
        super(Gemp, self).__init__()
        self.p = p
        self.eps = eps
        
    def forward(self, x):
        x = x.clamp(self.eps).pow(self.p)
        x = nn.functional.adaptive_avg_pool1d(x, 1).pow(1. / (self.p) )
        return x
    