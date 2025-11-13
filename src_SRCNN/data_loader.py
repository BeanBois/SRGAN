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
        SRCNN Dataset following SRGAN structure
        
        Args:
            img_dir: Directory containing training images
            scale_factor: Upscaling factor (2, 3, or 4)
            patch_size: Size of sub-images to extract (default: 33)
            stride: Stride for patch extraction (default: 14)
        """
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        
        # Get list of image files
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Pre-extract all patches
        self.lr_patches = []
        self.hr_patches = []
        
        print(f"Extracting patches from {len(self.img_files)} images...")
        for img_file in self.img_files:
            self._extract_patches_from_image(img_file)
        
        print(f"Total patches extracted: {len(self.hr_patches)}")
    
    def _extract_patches_from_image(self, img_file):
        """Extract patches from a single image"""
        img_path = os.path.join(self.img_dir, img_file)
        
        # Read image and convert to YCbCr
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        
        # Use only Y channel (luminance)
        y_channel = img_ycrcb[:, :, 0]
        
        # Extract patches
        h, w = y_channel.shape
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                hr_patch = y_channel[i:i+self.patch_size, j:j+self.patch_size]
                
                # Create low-resolution version
                lr_patch = self._generate_lr_patch(hr_patch)
                
                self.hr_patches.append(hr_patch)
                self.lr_patches.append(lr_patch)
    
    def _generate_lr_patch(self, hr_patch):
        """Generate low-resolution patch from high-resolution patch"""
        # Downsample
        h, w = hr_patch.shape
        lr_size = (w // self.scale_factor, h // self.scale_factor)
        lr_patch = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_CUBIC)
        
        # Upsample back to original size (bicubic interpolation)
        lr_patch = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return lr_patch
    
    def __len__(self):
        return len(self.hr_patches)
    
    def __getitem__(self, idx):
        """Return a single patch pair"""
        lr_patch = self.lr_patches[idx].astype(np.float32) / 255.0
        hr_patch = self.hr_patches[idx].astype(np.float32) / 255.0
        
        # Convert to PyTorch tensors [C, H, W]
        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)  # Add channel dimension
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        
        return lr_patch, hr_patch