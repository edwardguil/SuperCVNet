from .base import AbstractVectorDB
import torch

class TensorVectorDB(AbstractVectorDB):
    def __init__(self, vectors:torch.Tensor, labels:torch.Tensor):
        self.vectors = vectors
        self.labels = labels

    def get_top_k(self, input:torch.Tensor, k: int):
        # Input is of shape batch_size x feature_dim
        # Vectors is of shape num_vectors x feature_dim
        input_unsqueezed = input.unsqueeze(1)  # shape becomes [batch_size, 1, feature_dim]
        vectors_unsqueezed = self.vectors.unsqueeze(0)  # shape becomes [1, num_vectors, feature_dim]
        cosine_similarity_matrix = torch.cosine_similarity(input_unsqueezed, vectors_unsqueezed, dim=-1)
        top_k_scores, top_k_indices = torch.topk(cosine_similarity_matrix, k=k, dim=-1)
        return self.vectors[top_k_indices], top_k_scores, top_k_indices