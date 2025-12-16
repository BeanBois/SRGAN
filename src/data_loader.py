import torch 
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import torchvision.transforms as T

def downscale_image(img, r=4, sigma=None):
    """
    Downscale image following SRGAN paper procedure:
    1. Apply Gaussian blur
    2. Downsample by factor r using bicubic interpolation
    
    Args:
        img: Input image tensor in [-1, 1] range OR PIL Image
        r: Downsampling factor (default: 4)
        sigma: Gaussian blur sigma (default: r/2)
    
    Returns:
        Downscaled image tensor in [-1, 1] range
    """
    if sigma is None:
        sigma = r / 2.0
    
    if isinstance(img, torch.Tensor):
        # Denormalize from [-1,1] to [0,1] and convert to PIL
        img_numpy = ((img + 1.0) / 2.0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray((img_numpy * 255).astype('uint8'))
    else:
        img_pil = img
    
    # Apply Gaussian blur
    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    # Downsample
    new_size = (int(img_pil.width / r), int(img_pil.height / r))
    downsampled = blurred.resize(new_size, Image.BICUBIC)
    
    # Convert back to tensor and normalize to [-1, 1]
    to_tensor = T.ToTensor()
    lr_tensor = to_tensor(downsampled)  # [0, 1]
    lr_tensor = lr_tensor * 2.0 - 1.0    # [-1, 1]
    
    return lr_tensor

class WholeImageDataset(Dataset):
    """Dataset that loads whole images without cropping for evaluation."""
    def __init__(self, img_dir, downscale_factor=4):
        self.img_dir = img_dir
        self.downscale_factor = downscale_factor
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        hr_image_pil = Image.open(img_path).convert('RGB')
        
        # Convert to tensor [0, 1]
        hr_image = T.ToTensor()(hr_image_pil)
        
        # Scale HR to [-1, 1] to match training
        hr_image = hr_image * 2.0 - 1.0
        
        # Create LR image (also in [-1, 1])
        lr_image = downscale_image(hr_image, r=self.downscale_factor)
        
        return lr_image, hr_image, img_name

class ImgDataset(Dataset):
    """
    Dataset for SRGAN training following the paper:
    - Each mini-batch contains patches from DISTINCT images (not multiple patches from same image)
    - One 96x96 patch randomly cropped from each image per epoch
    - Both LR and HR images normalized to [-1, 1] range
    """
    def __init__(self, img_dir, hr_size=96, downscale_factor=4, is_training=True):
        self.img_dir = img_dir
        self.hr_size = hr_size
        self.downscale_factor = downscale_factor
        self.is_training = is_training
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.img_files)} images in {img_dir}")
        
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
        # Each image appears once per epoch
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        hr_image_pil = Image.open(img_path).convert('RGB')
        
        if self.is_training:
            # Check if image is large enough for random crop
            if hr_image_pil.width < self.hr_size or hr_image_pil.height < self.hr_size:
                # Resize if too small
                new_size = (max(hr_image_pil.width, self.hr_size + 10), 
                           max(hr_image_pil.height, self.hr_size + 10))
                hr_image_pil = hr_image_pil.resize(new_size, Image.BICUBIC)
        
        # Apply transforms to get random/center crop
        hr_image = self.hr_transform(hr_image_pil)  # [0, 1]
        
        # Scale HR to [-1, 1] as per paper
        hr_image = hr_image * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        # Create LR image (also in [-1, 1] range)
        lr_image = downscale_image(hr_image, r=self.downscale_factor)
        
        return lr_image, hr_image