import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os
import random
import numpy as np
from torchvision.transforms import ToTensor

class MyCustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, seed = 1234):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        self.data = [line.strip().split(',') for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(label)

        if self.transform:
            image = self.transform(image)

        return image, label
    
class DegrationDataset(Dataset):
    def __init__(self, args, train = True, image_dir = "data/hw4_realse_dataset/train/degraded", seed = 1234):
        self.image_dir = image_dir
        self.train = train
        random.seed(seed)
        indexes = list(range(1, 1601))
        rain_data = [image_dir + f"/rain-{i}.png" for i in indexes]
        snow_data = [image_dir + f"/snow-{i}.png" for i in indexes]
        random.shuffle(rain_data)
        random.shuffle(snow_data)
        rain_data = [[0, path] for path in rain_data]
        snow_data = [[1, path] for path in snow_data]
        self.data = rain_data[100:] + snow_data[100:] if train else rain_data[:100] + snow_data[:100]
        self.ToTensor = ToTensor()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        de_type, img_name = self.data[idx]
        cln_name = self._get_nonhazy_name(img_name)
        dgrad_img = Image.open(img_name).convert('RGB')
        clean_img = Image.open(cln_name).convert('RGB')

        if(self.train and random.random()<0.5):
            dgrad_img = ImageOps.mirror(dgrad_img)
            clean_img = ImageOps.mirror(clean_img)

        dgrad_img = self.ToTensor(np.array(dgrad_img))
        clean_img = self.ToTensor(np.array(clean_img))

        return [cln_name, de_type], dgrad_img, clean_img
    
    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("degraded")[0] + 'clean/'
        name = hazy_name.split('/')[-1].split('-')[0] + "_clean-" + hazy_name.split('/')[-1].split('-')[1]
        nonhazy_name = dir_name + name
        return nonhazy_name
