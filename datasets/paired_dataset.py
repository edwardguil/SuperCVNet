import random
from torch.utils.data import Dataset
from collections import defaultdict
import os, json

class PairedDataset(Dataset):
    def __init__(self, dataset, root='./data', name=None):
        self.dataset = dataset
        self.label_to_indices = defaultdict(list)
        base = os.path.join(root, 'pairings')
        name = name if name else dataset.__class__.__name__

        if not os.path.exists(base):
            os.makedirs(base)

        file = os.path.join(base, f'{name}_pairings.json')
        
        if os.path.exists(file):
            print("Using cache found for dataset pairings")
            with open(file, "r+") as f:
                self.label_to_indices = json.load(f)

            self.label_to_indices = {int(k): v for k, v in self.label_to_indices.items()}
        else:
            print("Generating list of dataset pairings")
            for index, (_, label) in enumerate(self.dataset):
                self.label_to_indices[label].append(index)

            with open(file, "w+") as f:
                json.dump(self.label_to_indices, f)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # Randomly choosing a positive sample index and fetching the sample
        positive_index = random.choice(self.label_to_indices[label])
        positive_img, _ = self.dataset[positive_index]
        
        return img, positive_img, label