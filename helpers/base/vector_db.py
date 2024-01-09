from abc import ABC, abstractmethod
from torch import Tensor
from typing import Tuple

class AbstractVectorDB(ABC):
    """Abstract base class for a vector database.

    This class defines the interface for a vector database. Subclasses should implement
    the `get_top_k` method to return the top k most similar vectors to a given input vector/s.

    Methods:
        get_top_k: Given an input vector and a parameter k, returns the top k most similar vectors in the database,
                   along with their cosine similarity scores and indices.
    """

    @abstractmethod
    def get_top_k(self, input: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Abstract method to get the top k most similar vectors to the input vector/s.
        This function should be able to handle both single and batched inputs.

        Args:
            input (Tensor): The input vector of shape [batch_size, feature_dim].
            k (int): The number of most similar vectors to return.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing three tensors: 
                - The top k most similar vectors of shape [batch_size, k, feature_dim], 
                - Their cosine similarity scores of shape [batch_size, k], 
                - Their indices/ids in the database of shape [batch_size, k].
        """
        pass