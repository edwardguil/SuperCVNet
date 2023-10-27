import torch
import torch.nn.functional as F
import torch.nn as nn

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    


# written by Shihao Shao (shaoshihao@pku.edu.cn)
class Gemp(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, p=4.6, eps = 1e-8):
        super(Gemp, self).__init__()
        self.p = p
        self.eps = eps
        
    def forward(self, x):
        x = x.clamp(self.eps).pow(self.p)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).pow(1. / (self.p) )
        return x

# written by Shihao Shao (shaoshihao@pku.edu.cn)
class Relup(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, alpha=0.014):
        super(Relup, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        x = x.clamp(self.alpha)
        return x
    
# written by Shihao Shao (shaoshihao@pku.edu.cn)
class Rgem(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, pr=2.5, size = 5):
        super(Rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size-1)//2.))
    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x = 0.5*self.lppool(self.pad(x/nominater)) + 0.5*x
        return x
    
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