import torch

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

