#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:43:24 2025

@author: sushmitachandel
"""
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom dataset class to load images and ground truths
class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
            )
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.input_dir, self.image_files[idx])
        label_path = os.path.join(self.output_dir, self.image_files[idx].replace('.png','.npy'))
        image = Image.open(img_path).convert('L')
        label = np.load(label_path)

        if self.transform:
            image = self.transform(image)

        # Convert ground truth to tensor
        label = torch.tensor(label, dtype=torch.float32)
        label_new = label.permute(2, 0, 1).float()
        # print("Hi Max value:", label_new.max().item())
        # print("Min value:", label_new.min().item())
        # print("Mean value:", label_new.mean().item())
        # Chnage the label image by adding background class as a seperate class
        background_mask = (label_new.sum(dim=0) == 0).float()
        background_channel = background_mask.unsqueeze(0)
        label_new = torch.cat([label_new,background_channel], dim = 0)

        return image, label_new
    
def get_dataloaders(train_path_input,val_path_input,train_path_output,val_path_output,
                    batch_size=128):
    
    # Transformations for training dataset
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.2289],
                         [0.0396])
        ])

    # Transformations for testing/validation dataset
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.2289],
                         [0.0396])
        ])  
    
    train_dataset = ImageDataset(train_path_input, train_path_output, transform=train_transforms)
    val_dataset = ImageDataset(val_path_input, val_path_output, transform=val_transforms)
    # print(len(train_dataset))
    # print(len(val_dataset))

    # # DataLoaders
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    image, ground_truths = next(iter(train_loader))
    # print(ground_truths.shape)
    num_classes = ground_truths.shape[1]
    print("Number of classes:", num_classes)
    print("Shape of the image is (C,H,W) where C donotes number of channels:", image.shape[1:4])
    print("Shape of the ground truth image is (C,H,W) where C denotes no. of classes:", ground_truths.shape[1:4])
    print("Batch size is:", ground_truths.shape[0])
    classes = [i for i in range(num_classes)]
        
    len_train_dataset = len(train_loader.dataset)
    len_val_dataset = len(val_loader.dataset)
    
    return train_loader, val_loader, classes


# import numpy as np
# import matplotlib.pyplot as plt

# train_path_input = '/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/input/'
# val_path_input = '/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/input/'
# train_path_output = '/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/output/'
# val_path_output = '/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/output/'

# train_loader, val_loader, classes = get_dataloaders(train_path_input,val_path_input,
#                                            train_path_output,val_path_output)
# # get_dataloaders(train_path_input,val_path_input,train_path_output,val_path_output)










