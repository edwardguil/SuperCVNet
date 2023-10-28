import random
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
import os, json, tarfile, hashlib
from multiprocessing import Process
from urllib.request import urlretrieve
from torchvision.io import read_image

class GoogleLandMarkDataset(Dataset):
    def __init__(self, root='./data', download=True, verify=True, split='train', num_proc=8, remove_tars=True, transform=None):
        self.root = os.path.join(root, 'google-landmarks')
        self.num_proc = num_proc
        self.remove_tars = remove_tars
        self.metadata = os.path.join(self.root, 'metadata')
        self.split = split
        self.limit = 499 if self.split == 'train' else 19 if self.split == 'test' else 99

        # Ensure correct parameter has been passsed
        if self.split not in ['train', 'test', 'split']:
            raise ValueError("split must be: 'train', 'test' or 'index'")
        
        # Train set uses different structure than test and index
        self.train = self.split == 'train'
        self.csv = 'train_clean.csv' if self.train else f'{self.split}.csv'

        # Create base folder e.g. {root}/google-landmarks/{split}/ if it dosent exist
        if not os.path.exists(os.path.join(self.root, split)):
            os.makedirs(os.path.join(self.root, split))

        if download:
            self.download()
            
        if os.path.exists(os.path.join(self.metadata, self.csv)):
            self.data = pd.read_csv(os.path.join(self.metadata, self.csv))
        
        if verify:
            self.verify()

        self.image_ids = []
        for _, row in self.data.iterrows():
            for img_id in row['images'].split():
                self.image_ids.append((img_id, row['landmark_id']))

    def download_check_and_extract(self, start, end, split):
        def check_hash(images_md5_path, images_file_path):
            if os.path.exists(images_file_path):
                hash_md5 = hashlib.md5()
                with open(images_file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                calculated_md5 = hash_md5.hexdigest()
                
            hash_md5 = hashlib.md5()
            with open(images_file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            calculated_md5 = hash_md5.hexdigest()

            # Read the provided md5 checksum
            with open(images_md5_path, "r") as file:
                provided_md5 = file.readline().strip().split()[0]

            # Validate the md5 checksum of the downloaded file
            if calculated_md5 == provided_md5:
                with tarfile.open(images_file_path) as tar:
                    tar.extractall(path=os.path.join(self.root, self.split))
                    if self.remove_tars:
                        os.remove(images_file_path)
                        os.remove(images_md5_path)
                return True
            return False

        for i in range(start, end):
            images_file_name = f"images_{i:03d}.tar"
            images_md5_file_name = f"md5.images_{i:03d}.txt"

            images_file_path = os.path.join(self.root, 'temp', images_file_name)
            images_md5_path = os.path.join(self.root, 'temp', images_md5_file_name)

            images_md5_url = f"https://s3.amazonaws.com/google-landmark/md5sum/{split}/{images_md5_file_name}"
            urlretrieve(images_md5_url, images_md5_path)

            if not (os.path.exists(images_file_path) and check_hash(images_md5_path, images_file_path)):
                images_tar_url = f"https://s3.amazonaws.com/google-landmark/{split}/{images_file_name}"
                print("Downloading: ", images_file_name)
                urlretrieve(images_tar_url, images_file_path)
                if not check_hash(images_md5_path, images_file_path):
                    print(f"Downloading: {images_file_name} failed. Wrong hash.")
                
    def download(self):
        if not os.path.exists(self.metadata):
            os.makedirs(self.metadata)
            if not os.path.exists(os.path.join(self.metadata, self.csv)):
                urlretrieve(f"https://s3.amazonaws.com/google-landmark/metadata/{self.csv}", os.path.join(self.metadata, self.csv))
        
        if not os.path.exists(os.path.join(self.root, 'temp')):
            os.makedirs(os.path.join(self.root, 'temp'))

        processes = []
        each = self.limit // self.num_proc
        for i in range(0, self.num_proc):
            start = each * i
            end = self.limit + 1 if i == self.num_proc-1 else start + each
            p = Process(target=self.download_check_and_extract, args=(start, end, self.split))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

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
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get the image_id and landmark_id using the provided index
        img_id, landmark_id = self.image_ids[index]

        # Construct the image path based on the ID
        img_path = os.path.join(self.root, self.split, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
        
        # Read the image
        image = read_image(img_path)

        if self.transform is not None:
            image = self.transform(image)
        
        # Return the image and landmark_id
        return image, landmark_id
    

class DatasetPaired(Dataset):
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