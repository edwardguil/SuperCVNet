import torch
import torch.nn.functional as F
from .curricularface import CurricularFace

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
            # If not enough space, began overwriting oldest data
            remain_space = self.queue_size - self.index

            # Add elements to the end of the array (to make it full)
            self.queue[self.index:self.index+remain_space, :] = tensor[0:remain_space]
            self.labels[self.index:self.index+remain_space] = labels[0:remain_space]

            # Reset the index
            self.index = 0

            # Start adding elements to start of the array (oldest elements)
            self.queue[self.index:batch_size-remain_space, :] = tensor[remain_space:batch_size]
            self.labels[self.index:batch_size-remain_space] = labels[remain_space:batch_size]

            # Update index
            self.index = batch_size - remain_space
        else:
            # If enough 'space' since last reset/init, just add/overwrite
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


    