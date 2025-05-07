
import os
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import skimage.io as sio
import torch
from torch.utils.data import Dataset

class SegmentationTransform:
    def __init__(
        self,
        crop_size=(0.2, 0.2),
        resize_size=(512, 512),
        hflip_prob=0.5,
        vflip_prob=0.5,
        color_jitter_params=(0.15, 0.15, 0.1, 0.05),
        val = False
    ):
        self.val = val
        self.crop_w,  self.crop_h= crop_size
        self.resize_size = resize_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.brightness, self.contrast, self.saturation, self.hue = color_jitter_params

    def __call__(self, image, instance_map, class_map):
        if(self.val):
            image = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
            return image, instance_map, class_map
        image = Image.fromarray(image.astype(np.uint8))
        instance_map = Image.fromarray(instance_map.astype(np.uint8), mode="L")
        class_map = Image.fromarray(class_map.astype(np.uint8), mode="L")

        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            instance_map = instance_map.transpose(Image.FLIP_LEFT_RIGHT)
            class_map = class_map.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < self.vflip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            instance_map = instance_map.transpose(Image.FLIP_TOP_BOTTOM)
            class_map = class_map.transpose(Image.FLIP_TOP_BOTTOM)

        image, instance_map, class_map = self.random_crop(image, instance_map, class_map)
        image = self.apply_color_jitter(image)
        image = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
        instance_map = np.array(instance_map, dtype=np.uint16)
        class_map = np.array(class_map, dtype=np.uint8)

        return image, instance_map, class_map

    def random_crop(self, image, instance_map, class_map):
        w, h = image.size

        scale_w = random.uniform(self.crop_w, 1.0)
        scale_h = random.uniform(self.crop_h, 1.0)
        crop_w = int(w * scale_w)
        crop_h = int(h * scale_h)

        max_x = w - crop_w
        max_y = h - crop_h
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)

        cropped_image = image.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        cropped_instance = instance_map.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        cropped_class = class_map.crop((x1, y1, x1 + crop_w, y1 + crop_h))

        return cropped_image, cropped_instance, cropped_class

    def apply_color_jitter(self, image):
        factor = 1 + (random.random() - 0.5) * 2 * self.brightness
        image = ImageEnhance.Brightness(image).enhance(factor)

        factor = 1 + (random.random() - 0.5) * 2 * self.contrast
        image = ImageEnhance.Contrast(image).enhance(factor)

        factor = 1 + (random.random() - 0.5) * 2 * self.saturation
        image = ImageEnhance.Color(image).enhance(factor)

        h = image.convert('HSV')
        np_hsv = np.array(h)
        shift = int((random.random() - 0.5) * 2 * self.hue * 255)
        np_hsv[..., 0] = (np_hsv[..., 0].astype(int) + shift) % 256
        image = Image.fromarray(np_hsv, mode='HSV').convert('RGB')

        return image
    
class CoNSePDataset(Dataset):
    def __init__(self, image_dir, transforms=None, nclass = 4, seed = 20250501, val_ratio = 0.1, train = True):
        self.image_dir = image_dir
        self.transforms = transforms
        all_files = sorted(os.listdir(image_dir))
        random.Random(seed).shuffle(all_files)

        split = int(len(all_files) * (1 - val_ratio))
        if train:
            self.image_files = all_files[:split]
        else:
            self.image_files = all_files[split:]

        self.nclass = nclass
        self.transforms = transforms

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        folder_path = os.path.join(self.image_dir, img_name)
        img_path = os.path.join(folder_path, "image.tif")

        image = cv2.imread(img_path)
        instance_map = np.zeros(image.shape[:2], dtype=np.uint8)
        class_map = np.zeros(image.shape[:2], dtype=np.uint8)

        for i in range(1, self.nclass + 1):
            instance_path = Path(os.path.join(folder_path, f"class{i}.tif"))
            if not instance_path.exists():
                continue

            tem_map = sio.imread(str(instance_path))
            instance_map[tem_map > 0] = tem_map[tem_map > 0]
            class_map[tem_map > 0] = i

        if self.transforms:
            image, instance_map, class_map = self.transforms(image, instance_map, class_map)

        obj_ids = np.unique(instance_map)
        obj_ids = obj_ids[obj_ids != 0]
        masks = instance_map == obj_ids[:, None, None]

        boxes = []
        labels = []
        areas = []

        for i, obj_id in enumerate(obj_ids):
            mask = masks[i]
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if(xmin==xmax or ymin==ymax):
                continue
            boxes.append([xmin, ymin, xmax, ymax])

            cls = class_map[instance_map == obj_id][0]
            labels.append(int(cls))
            areas.append(mask.sum().item())

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(masks, dtype=torch.uint8),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_id": idx
        }

        return torch.tensor(image), target

    def __len__(self):
        return len(self.image_files)


