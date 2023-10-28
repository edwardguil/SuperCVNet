import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from layers import GeM, Gemp, Rgem, Sgem, Relup

class GlobalNetwork(torch.nn.Module):
    def __init__(self, reduction_dim=2048, resnet_depth=50):
        super(GlobalNetwork, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', f'resnet{resnet_depth}', weights='DEFAULT')
        # Backbone is Resnet: stem (conv1, bn1, relu, maxpool) + first four blocks
        self.stem = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.gem_pooling = GeM()
        self.whitening = nn.Linear(2048, reduction_dim, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gem_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.whitening(x)

        return x

class MomentumNetwork(GlobalNetwork):
    def __init__(self, reduction_dim=2048, momentum=0.999):
        super().__init__(reduction_dim)
        self.momentum = momentum

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].data.copy_(self.momentum * own_state[name].data + (1 - self.momentum) * param.data)

class CVNetGlobal(torch.nn.Module):
    def __init__(self, reduction_dim=2048, momentum=0.999):
        super(CVNetGlobal, self).__init__()
        self.global_network = GlobalNetwork(reduction_dim)
        self.momentum_network = MomentumNetwork(reduction_dim, momentum)

    def forward(self, x, x_positive, with_momentum=True):
        global_features = self.global_network(x)
        if with_momentum:
            with torch.no_grad():  # no gradient to momentum features
                momentum_features = self.momentum_network(x_positive)
            return global_features, momentum_features
        else:
            return global_features
        
class SuperGlobalNetwork(GlobalNetwork):
    '''Don't include RelUP as only %1 increase in performance'''
    def __init__(self, reduction_dim=2048, resnet_depth=50):
        super().__init__(reduction_dim, resnet_depth)
        self.rgem = Rgem()
        self.gemp = Gemp()
        self.sgem = Sgem()

    def _forward_singlescale(self, x, gemp=True, rgem=True):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if rgem:
            x = self.rgem(x)

        x = x.view(x.shape[0], x.shape[1], -1) 
        
        if gemp:
            x = self.gemp(x)

        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=-1)
        x = self.whitening(x)   

        return x

    def forward(self, x, scale=3, gemp=True, rgem=True, sgem=True):
        feature_list = []
        scales = [0.5, 0.7071, 1., 1.4142, 2.] if scale == 5 else ([0.7071, 1., 1.4142] if scale == 3 else [1.])
        
        for scl in scales:
            x_rescaled = TF.resize(x, [int(x.shape[-2]*scl), int(x.shape[-1]*scl)])
            features = self._forward_singlescale(x_rescaled, gemp, rgem)
            feature_list.append(features)
            
        if sgem:
            out_features = self.sgem(feature_list)
        else:
            out_features = torch.mean(torch.stack(feature_list, 0), 0)
            
        return out_features


class SuperMomentumNetwork(SuperGlobalNetwork):
    def __init__(self, reduction_dim=2048, momentum=0.999):
        super().__init__(reduction_dim)
        self.momentum = momentum

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].data.copy_(self.momentum * own_state[name].data + (1 - self.momentum) * param.data)


class SuperCVNetGlobal(torch.nn.Module):
    def __init__(self, reduction_dim=2048, momentum=0.999):
        super(SuperCVNetGlobal, self).__init__()
        self.global_network = SuperGlobalNetwork(reduction_dim)
        self.momentum_network = SuperMomentumNetwork(reduction_dim, momentum)

    def forward(self, x, x_positive, with_momentum=True):
        global_features = self.global_network(x)
        if with_momentum:
            with torch.no_grad():  # no gradient to momentum features
                momentum_features = self.momentum_network(x_positive)
            return global_features, momentum_features
        else:
            return global_features
