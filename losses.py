import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

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
        loss = -torch.log((numerator / denominator) + epsilon)        
        
        return loss.mean()

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
        loss = -torch.log((numerator / denominator) + epsilon).mean()

        return loss
    


class  CurricularFace(nn.Module):
    """Implementation of
    `CurricularFace: Adaptive Curriculum Learning\
        Loss for Deep Face Recognition`_.

    .. _CurricularFace\: Adaptive Curriculum Learning\
        Loss for Deep Face Recognition:
        https://arxiv.org/abs/2004.00288

    Official `pytorch implementation`_.

    .. _pytorch implementation:
        https://github.com/HuangYG123/CurricularFace

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = CurricularFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """  # noqa: RST215

    def __init__(  # noqa: D107
        self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5,
    ):
        super(CurricularFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))

        nn.init.normal_(self.weight, std=0.01)

    def __repr__(self) -> str:  # noqa: D105
        rep = (
            "CurricularFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"m={self.m},s={self.s}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor, label: torch.LongTensor = None) -> torch.Tensor:
            """
            Args:
                input: input features,
                    expected shapes ``BxF`` where ``B``
                    is batch dimension and ``F`` is an
                    input feature dimension.
                label: target classes,
                    expected shapes ``B`` where
                    ``B`` is batch dimension.
                    If `None` then will be returned
                    projection on centroids.
                    Default is `None`.

            Returns:
                tensor (logits) with shapes ``BxC``
                where ``C`` is a number of classes.
            """
            cos_theta = torch.mm(F.normalize(input), F.normalize(self.weight, dim=0))
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

            if label is None:
                return cos_theta

            target_logit = cos_theta[torch.arange(0, input.size(0)), label].view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(
                target_logit > self.threshold, cos_theta_m, target_logit - self.mm
            )

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t

            cos_theta[mask] = hard_example * (self.t + hard_example)
            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s

            return output
    
    
class TensorQueue(torch.nn.Module):
    def __init__(self, feature_dim=2048, queue_size=8192):
        super(TensorQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0 
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        self.register_buffer("labels", torch.empty(queue_size, dtype=torch.long))
        
    def enqueue(self, tensor, labels):
        batch_size = tensor.shape[0]
        if self.index + batch_size > self.queue_size:
            # If not enough space, overwrite the oldest data
            remain_space = self.queue_size - self.index
            self.queue[:, self.index:] = tensor[:, :remain_space]
            self.queue[:, :batch_size - remain_space] = tensor[:, remain_space:]
            self.labels[self.index:] = labels[:remain_space]
            self.labels[:batch_size - remain_space] = labels[remain_space:]
            self.index = batch_size - remain_space
        else:
            # If enough space, just add
            self.queue[self.index:self.index+batch_size, :] = tensor
            self.labels[self.index:self.index+batch_size] = labels
            self.index += batch_size
        
    def get_descriptors(self):
        return self.queue

    def get_queue(self):
        return self.queue, self.labels
    
    def find_samples(self, labels):     
        positives_mask = self.labels.unsqueeze(1) == labels.unsqueeze(0)
        negatives_mask = ~positives_mask
        return positives_mask, negatives_mask

