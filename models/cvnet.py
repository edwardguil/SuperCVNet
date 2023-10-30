import torch
import torch.nn as nn
from layers import GeM
from .base import CVLearner, Correlation

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

    def forward(self, x, ret_intermediate=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if ret_intermediate:
            # CVNET uses output from layer3 (s3) without RELU activation
            # layer3 in this version of ResNet has no final relu. So just return layer3
            return x
        
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

class CVNetRerank(torch.nn.Module):
    def __init__(self, input_dim = 1024):
        super(CVNetRerank, self).__init__()
        self.scales = [0.25, 0.5, 1.0]
        self.num_scales = len(self.scales)
        self.conv2ds = nn.ModuleList([nn.Conv2d(input_dim, 256, kernel_size=3, padding=1, bias=False) for _ in self.scales])
        self.cv_learner = CVLearner([self.num_scales**2 for _ in range(3)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query_features, key_features):
        # requires output from global network intemediate layer
        # i.e. GlobalNetwork.foward(query_image, ret_intermediate=True)
        corr = Correlation.build_crossscale_correlation(query_features, key_features, self.scales, self.conv2ds)
        logits = self.cv_learner(corr)
        score = self.softmax(logits)[:,1]
        return score
