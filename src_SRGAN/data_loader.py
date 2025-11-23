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
    def __init__(self, img_dir, hr_size=96, downscale_factor=4, target_transform=None, is_training = True):
        self.img_dir = img_dir
        self.hr_size = hr_size
        self.downscale_factor = downscale_factor
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.endswith('.png')])
        
        if is_training:
                self.hr_transform = T.Compose([
                T.RandomCrop(hr_size),
                T.RandomHorizontalFlip(p=0.5),  # ✅ Add this
                T.RandomVerticalFlip(p=0.5),    # ✅ Add this
                T.ToTensor(),  # [0, 1]
            ])
        else:

            self.hr_transform = T.Compose([
                T.CenterCrop(hr_size),
                T.ToTensor(),  # [0, 1]
            ])
        
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        hr_image_pil = Image.open(img_path).convert('RGB')
        hr_image = self.hr_transform(hr_image_pil)  # [0, 1]
        
        # Scale HR to [-1, 1] as per paper
        hr_image = hr_image * 2.0 - 1.0  # ✅ [0,1] -> [-1,1]
        
        lr_image = downscale_image(hr_image, r=self.downscale_factor)
        # lr_image is already in [0, 1] from ToTensor in downscale_image
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.img_files)
    