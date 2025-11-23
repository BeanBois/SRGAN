import torch 
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import torchvision.transforms as T

def downscale_image(img, r=4, sigma=None):
    if sigma is None:
        sigma = r / 2.0
    
    if isinstance(img, torch.Tensor):
        # Denormalize from [-1,1] to [0,255] and convert to PIL
        img_numpy = ((img + 1.0) / 2.0).clamp(0, 1).numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray((img_numpy * 255).astype('uint8'))
    else:
        img_pil = img
    
    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    new_size = (int(img_pil.width / r), int(img_pil.height / r))
    downsampled = blurred.resize(new_size, Image.BICUBIC)  # Changed to BICUBIC
    
    to_tensor = T.ToTensor()
    return to_tensor(downsampled)  # Returns [0, 1]


class ImgDataset(Dataset):
    def __init__(self, img_dir, hr_size=96, downscale_factor=4, 
                 patches_per_image=1, is_training=True):
        self.img_dir = img_dir
        self.hr_size = hr_size
        self.downscale_factor = downscale_factor
        self.patches_per_image = patches_per_image  # ✅ NEW
        self.is_training = is_training
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if is_training:
            self.hr_transform = T.Compose([
                T.RandomCrop(hr_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ToTensor(),  # [0, 1]
            ])
        else:
            self.hr_transform = T.Compose([
                T.CenterCrop(hr_size),
                T.ToTensor(),  # [0, 1]
            ])
    
    def __len__(self):
        # ✅ Multiply by patches_per_image
        return len(self.img_files) * self.patches_per_image
    
    def __getitem__(self, idx):
        # ✅ Map virtual idx to actual image file
        actual_img_idx = idx // self.patches_per_image
        img_name = self.img_files[actual_img_idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        hr_image_pil = Image.open(img_path).convert('RGB')
        
        # Check if image is large enough
        if hr_image_pil.width < self.hr_size or hr_image_pil.height < self.hr_size:
            # Resize if too small
            new_size = (max(hr_image_pil.width, self.hr_size + 10), 
                       max(hr_image_pil.height, self.hr_size + 10))
            hr_image_pil = hr_image_pil.resize(new_size, Image.BICUBIC)
        
        hr_image = self.hr_transform(hr_image_pil)  # [0, 1]
        
        # Scale HR to [-1, 1] as per paper
        hr_image = hr_image * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        lr_image = downscale_image(hr_image, r=self.downscale_factor)
        # lr_image is in [0, 1]
        
        return lr_image, hr_image