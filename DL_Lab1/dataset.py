from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch
import os
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, img_dir, preprocess = None):
        self.img_dir = img_dir
        self.imglabel_list = os.listdir(img_dir)
        self.img_list = []
        self.label_list = []
        self.weights = [0] * len(self.imglabel_list)
        self.preprocess = preprocess
        for label in self.imglabel_list:
            image_list = os.listdir(os.path.join(self.img_dir, label))
            num_label = int(label)
            self.weights[num_label] = len(image_list)
            for image in image_list:
                self.img_list.append(os.path.join(self.img_dir, label, image))
                self.label_list.append(num_label)
        self.weights = torch.tensor(self.weights, dtype=torch.float)
        self.weights = 1.0 / self.weights  
        self.weights = self.weights / self.weights.sum() 

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = read_image(self.img_list[idx])[:3, :, :]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if(self.preprocess is not None):
            image = transforms.functional.to_pil_image(image)
            image = self.preprocess(image)
        label = self.label_list[idx]
        return image, label
    
def dataloader(data_path, preprocess = None, batch_size=8, shuffle=True):
    dataset = MyDataset(data_path, preprocess = preprocess)
    return dataset.weights, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

Rval_preprocess = transforms.Compose([
    transforms.Resize(256),                  
    transforms.CenterCrop(224),             
    transforms.ToTensor(),                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

Rtrain_preprocess = transforms.Compose([
    transforms.Resize(256),                  
    transforms.CenterCrop(224),         
    transforms.RandomHorizontalFlip(),      
    transforms.RandomRotation(20),           
    transforms.ColorJitter(brightness=0.3, 
                           contrast=0.3, 
                           saturation=0.3, 
                           hue=0.2),         
    transforms.ToTensor(),                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

Preprocess = {"train": Rtrain_preprocess, "val": Rval_preprocess}