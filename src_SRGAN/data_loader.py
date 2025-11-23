import torch 
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import torchvision.transforms as T

# Gauss filter + downscaling
def downscale_image(img, r=4, sigma=None):
    if sigma is None:
        sigma = r / 2.0
    
    if isinstance(img, torch.Tensor):
        # Denormalize from [0,1] to [0,255] and convert to PIL
        img_pil = T.ToPILImage()(img)
    else:
        img_pil = img
    
    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    new_size = (int(img_pil.width / r), int(img_pil.height / r))
    downsampled = blurred.resize(new_size, Image.BILINEAR)
    
    to_tensor = T.ToTensor()
    return to_tensor(downsampled)


class ImgDataset(Dataset):
    def __init__(self, img_dir, hr_size=96, downscale_factor=4):
        self.img_dir = img_dir
        self.hr_size = hr_size
        self.downscale_factor = downscale_factor
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # HR images normalized to [-1, 1] as per paper
        self.hr_transform = T.Compose([
            T.RandomCrop(hr_size),  # Random crop for training
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Maps to [-1, 1]
        ])
        
        # LR images stay in [0, 1]
        self.lr_transform = T.ToTensor()
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        hr_image_pil = Image.open(img_path).convert('RGB')
        
        # Random crop first (same crop for both HR and LR)
        hr_image_pil = T.RandomCrop(self.hr_size)(hr_image_pil)
        
        # Create LR version
        lr_size = self.hr_size // self.downscale_factor
        lr_image_pil = hr_image_pil.filter(
            ImageFilter.GaussianBlur(radius=self.downscale_factor / 2.0)
        )
        lr_image_pil = lr_image_pil.resize((lr_size, lr_size), Image.BICUBIC)
        
        # Apply transforms
        lr_image = T.ToTensor()(lr_image_pil)  # [0, 1]
        hr_image = T.Normalize([0.5]*3, [0.5]*3)(T.ToTensor()(hr_image_pil))  # [-1, 1]
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.img_files)
    