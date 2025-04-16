import json
import os

from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

Digit_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
Annotation_key = ['id', 'image_id', 'bbox', 'category_id', 'area', 'iscrowd']
Images_key = ['id', 'file_name', 'height', 'width']

class DigitDataset(Dataset):
    def __init__(self, images_dir, json_path, resize_size = (256, 128), preprocess = None):
        self.images_dir = images_dir
        self.transforms = preprocess
        with open(json_path, 'r') as file:
            ann_json = json.load(file)
        self.image_dict = {img['id']: img for img in ann_json['images']}
        self.resize_size = resize_size
        
        self.annotation_dict = dict()
        for ann in ann_json['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotation_dict:
                self.annotation_dict[image_id] = []
            self.annotation_dict[image_id].append(ann)
        
        self.image_ids = list(self.image_dict)
        self.label_dict = {cat['id']: i for i, cat in enumerate(ann_json['categories'])}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_dict[image_id]
        annotations = self.annotation_dict.get(image_id, [])

        img_path = os.path.join(self.images_dir, image_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        new_w, new_h = self.resize_size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        img = F.resize(img, (new_h, new_w))

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann['bbox']

            nx = x * scale_x
            ny = y * scale_y
            nw = w * scale_x
            nh = h * scale_y

            boxes.append([nx, ny, nx + nw, ny + nh])
            labels.append(ann['category_id'])#self.label_dict[]
            areas.append(nw * nh)
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

def dataloader(data_path, data_json, preprocess = None, batch_size=8, shuffle=True):
    dataset = DigitDataset(data_path, data_json, preprocess = preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    return tuple(zip(*batch))

Rval_preprocess = T.Compose([      
    T.ToTensor(),
    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

Rtrain_preprocess = T.Compose([                 
    T.ColorJitter(  brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3, 
                    hue=0.2        ),         
    T.ToTensor(),
    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                   
])

Preprocess = {"train": Rtrain_preprocess,
               "valid": Rval_preprocess}