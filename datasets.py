import random
from torch.utils.data import Dataset
from collections import defaultdict
import os, json

class PairedCIFAR10(Dataset):
    def __init__(self, cifar_dataset):
        self.cifar_dataset = cifar_dataset
        self.label_to_indices = defaultdict(list)
        
        if os.path.exists("cache.json"):
            print("Using cache found for dataset pairings")
            with open("cache.json", "r+") as f:
                self.label_to_indices = json.load(f)

            self.label_to_indices = {int(k): v for k, v in self.label_to_indices.items()}
        else:
            for index, (_, label) in enumerate(self.cifar_dataset):
                self.label_to_indices[label].append(index)

            with open("cache.json", "w+") as f:
                json.dump(self.label_to_indices, f)
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, index):
        img, label = self.cifar_dataset[index]
        
        # Randomly choosing a positive sample index and fetching the sample
        positive_index = random.choice(self.label_to_indices[label])
        positive_img, _ = self.cifar_dataset[positive_index]
        
        return img, positive_img, label