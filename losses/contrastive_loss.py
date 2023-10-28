import torch
import torch.nn.functional as F
from .curricularface import CurricularFace
from helpers.classes import TensorQueue


class MomentumContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, feature_dim=2048, tau=0.05, s=64.0, m=0.5, queue_size=8192):
        super(MomentumContrastiveLoss, self).__init__()
        self.feature_dim = 2048
        self.queue_size = queue_size
        self.tau = tau
        self.curricular_face = CurricularFace(in_features=feature_dim, out_features=num_classes, s=s, m=m)  
        self.queue = TensorQueue(feature_dim=feature_dim, queue_size=queue_size)

    def clear_queue(self):
        self.queue = TensorQueue(feature_dim=self.feature_dim, queue_size=self.queue_size)

    def forward(self, dq_g, dp_g, labels):
        """
        dq_g: Tensor representing the query global descriptor
        dp_g: Tensor representing the momentum global descriptor 
        labels: Ground truth labels
        """
        # Enqueue the new momentum global descriptors
        self.queue.enqueue(dp_g, labels)

        # Compute the cosine similarity between dq_g and all descriptors in the queue
        cosine_similarity_matrix = torch.mm(F.normalize(dq_g), F.normalize(self.queue.get_descriptors()).t())

        # Applying CurricularFace margined cosine similarity only for the positive samples
        cf_similarity = self.curricular_face(dq_g, labels)  # It should return a vector of similarities
        
        # Replace the similarities of positive samples in the cosine_similarity_matrix with curricular_face similarities
        cosine_similarity_matrix[range(len(labels)), labels] = cf_similarity[range(len(labels)), labels]

        # Get the positive and negative samples based on labels
        positives, negatives = self.queue.find_samples(labels)

        # Compute the loss according to the equation
        numerator = torch.exp(cosine_similarity_matrix[positives.t()] / self.tau)
        denominator = numerator + torch.sum(torch.exp(cosine_similarity_matrix[negatives.t()] / self.tau), dim=0)
        
        # Add epislon to stop 0 values in numerator leading to inf values in log
        epsilon = 1e-10
        loss = -torch.log((numerator + epsilon) / (denominator + epsilon)).mean()

        return loss
    