import torch
from .curricularface import CurricularFace

class ClassificationLoss(torch.nn.Module):
    def __init__(self, num_classes, feature_dim=2048, tau=0.05, s=64.0, m=0.5):
        super(ClassificationLoss, self).__init__()
        self.tau = tau
        self.curricular_face = CurricularFace(in_features=feature_dim, out_features=num_classes, s=s, m=m)
        
    def forward(self, dq_g, labels):
        """
        dq_g: Tensor of shape (batch_size, feature_dim) - The output from the global network
        labels: Tensor of shape (batch_size) - Ground-truth labels
        """
        # Getting the curricular face margined cosine similarity
        margined_cosine_similarity = self.curricular_face(dq_g, labels)
        numerator = torch.exp(margined_cosine_similarity[range(len(labels)), labels] / self.tau)
        denominator = torch.sum(torch.exp(margined_cosine_similarity / self.tau), dim=1)
        
        # Add epislon to stop 0 values in numerator leading to inf values in log
        epsilon = 1e-10
        loss = -torch.log((numerator + epsilon) / (denominator + epsilon))        
        
        return loss.mean()