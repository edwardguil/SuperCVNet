import random
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
import os, json, subprocess, tarfile, hashlib
from urllib.request import urlretrieve
from torchvision.io import read_image

class GoogleLandMarkDataset(Dataset):
    def __init__(self, root='./data', download=True, verify=True, split='train'):
        self.root = os.path.join(root, 'google-landmarks')
        self.metadata = os.path.join(self.root, 'metadata')
        self.split = split
        self.limit = 499 if self.split == 'train' else 19 if self.split == 'test' else 99

        # Ensure correct parameter has been passsed
        if self.split not in ['train', 'test', 'split']:
            raise ValueError("split must be: 'train', 'test' or 'index'")

        # Create base folder e.g. {root}/google-landmarks/split/ if it dosent exist
        if not os.path.exists(os.path.join(self.root, split)):
            os.makedirs(os.path.join(self.root, split))

        if download:
            self.download()
        
        if verify:
            self.verify()

        self.labels_csv = os.path.join(self.metadata, f'{self.split}_clean.csv')
        if os.path.exists(self.labels_csv):
            self.data = pd.read_csv(self.labels_csv)

        
    def download(self):
        if not os.path.exists(self.metadata):
            os.makedirs(self.metadata)
            if not os.path.exists(os.path.join(self.metadata, f'{self.split}.csv')):
                urlretrieve(f"https://s3.amazonaws.com/google-landmark/metadata/{self.split}.csv", os.path.join(self.metadata, f'{self.split}.csv'))

        def download_check_and_extract(i, split):
            images_file_name = f"images_{i:03d}.tar"
            images_md5_file_name = f"md5.images_{i:03d}.txt"
            images_tar_url = f"https://s3.amazonaws.com/google-landmark/{split}/{images_file_name}"
            images_md5_url = f"https://s3.amazonaws.com/google-landmark/md5sum/{split}/{images_md5_file_name}"

            urlretrieve(images_tar_url, images_file_name)
            urlretrieve(images_md5_url, images_md5_file_name)

            hash_md5 = hashlib.md5()
            with open(images_file_name, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            calculated_md5 = hash_md5.hexdigest()

            # Read the provided md5 checksum
            with open(images_md5_file_name, "r") as file:
                provided_md5 = file.readline().strip().split()[0]

            # Validate the md5 checksum of the downloaded file
            if calculated_md5 == provided_md5:
                with tarfile.open(images_file_name) as tar:
                    tar.extractall(path=self.root)
                    print(f"{images_file_name} extracted!")
            else:
                print(f"MD5 checksum for {images_file_name} did not match checksum in {images_md5_file_name}")

        for i in range(0, self.limit):
             download_check_and_extract(i, self.split)

    def verify(self):
        for _, row in self.data.iterrows():
            image_ids = row['images'].split()
            for img_id in image_ids:
                img_path = os.path.join(self.root, self.split, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
                if not os.path.exists(img_path):
                    missing_images = True
                    break

        if missing_images:
            print(f"Warning: {len(missing_images)} images missing!")
        else:
            print("All images are present!")

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass



class PairedCIFAR10(Dataset):
    def __init__(self, cifar_dataset, file="./data/cifar10_pair_cache.json"):
        self.cifar_dataset = cifar_dataset
        self.label_to_indices = defaultdict(list)
        
        if os.path.exists(file):
            print("Using cache found for dataset pairings")
            with open(file, "r+") as f:
                self.label_to_indices = json.load(f)

            self.label_to_indices = {int(k): v for k, v in self.label_to_indices.items()}
        else:
            for index, (_, label) in enumerate(self.cifar_dataset):
                self.label_to_indices[label].append(index)

            with open(file, "w+") as f:
                json.dump(self.label_to_indices, f)
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, index):
        img, label = self.cifar_dataset[index]
        
        # Randomly choosing a positive sample index and fetching the sample
        positive_index = random.choice(self.label_to_indices[label])
        positive_img, _ = self.cifar_dataset[positive_index]
        
        return img, positive_img, label
    

dataset = GoogleLandMarkDataset()