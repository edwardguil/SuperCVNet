import torch.nn as nn

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