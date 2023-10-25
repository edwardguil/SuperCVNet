import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet
from layers import GeM

class GlobalNetwork(torch.nn.Module):
    def __init__(self, num_classes, feature_dim=2048, resnet_depth=50, reduction_dim=256):
        super(GlobalNetwork, self).__init__()
        self.resnet = ResNet(resnet_depth, reduction_dim)

        self.backbone = nn.Sequential( 
            self.resnet.stem,
            self.resnet.s1,
            self.resnet.s2,
            self.resnet.s3,
            self.resnet.s4
        )
        self.gem_pooling = GeM()
        self.whitening = nn.Linear(feature_dim, feature_dim, bias=False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gem_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.whitening(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class MomentumNetwork(GlobalNetwork):
    def __init__(self, num_classes, feature_dim=2048, momentum=0.999):
        super().__init__(num_classes, feature_dim)
        self.momentum = momentum

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].data.copy_(self.momentum * own_state[name].data + (1 - self.momentum) * param.data)

class CVNetGlobal(torch.nn.Module):
    def __init__(self, num_classes, feature_dim=2048):
        super(CVNetGlobal, self).__init__()
        self.global_network = GlobalNetwork(num_classes, feature_dim)
        self.momentum_network = MomentumNetwork(num_classes, feature_dim)

    def forward(self, x, x_positive, with_momentum=True):
        global_features = self.global_network(x)
        if with_momentum:
            with torch.no_grad():  # no gradient to momentum features
                momentum_features = self.momentum_network(x_positive)
            return global_features, momentum_features
        else:
            return global_features
