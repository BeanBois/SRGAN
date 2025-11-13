import torch 
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2

class SRDataset(Dataset):
    def __init__(self, img_dir, scale_factor=3, patch_size=33, stride=14):
        """
        Memory-efficient SRCNN Dataset - extracts patches on-the-fly
        """
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        
        # Get list of image files
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Calculate patch indices for each image
        self.patch_info = []  # [(img_idx, i, j), ...]
        
        print(f"Calculating patch locations from {len(self.img_files)} images...")
        for img_idx, img_file in enumerate(self.img_files):
            img_path = os.path.join(self.img_dir, img_file)
            img_pil = Image.open(img_path).convert('RGB')
            h, w = img_pil.size[1], img_pil.size[0]
            
            # Calculate valid patch positions
            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    self.patch_info.append((img_idx, i, j))
        
        print(f"Total patches: {len(self.patch_info)}")
    
    def __len__(self):
        return len(self.patch_info)
    
    def __getitem__(self, idx):
        """Extract and return a single patch pair on-the-fly"""
        img_idx, i, j = self.patch_info[idx]
        img_file = self.img_files[img_idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        # Read image and convert to YCbCr
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        
        # Extract Y channel patch
        y_channel = img_ycrcb[:, :, 0]
        hr_patch = y_channel[i:i+self.patch_size, j:j+self.patch_size]
        
        # Generate LR patch
        lr_patch = self._generate_lr_patch(hr_patch)
        
        # Normalize and convert to tensors
        lr_patch = lr_patch.astype(np.float32) / 255.0
        hr_patch = hr_patch.astype(np.float32) / 255.0
        
        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        
        return lr_patch, hr_patch
    
    def _generate_lr_patch(self, hr_patch):
        """Generate low-resolution patch from high-resolution patch"""
        h, w = hr_patch.shape
        lr_size = (w // self.scale_factor, h // self.scale_factor)
        lr_patch = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_CUBIC)
        lr_patch = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
        return lr_patch