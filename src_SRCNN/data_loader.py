import torch 
from torch.utils.data import Dataset, Sampler
import os
from PIL import Image
import numpy as np
import cv2
import random

class SRDataset(Dataset):
    def __init__(self, img_dir, scale_factor=3, patch_size=33, 
                 patches_per_image=100):
        """
        SRCNN Dataset with on-the-fly random patch extraction
        NO pre-extraction, NO caching - generates patches on demand!
        
        Args:
            img_dir: Directory containing training images
            scale_factor: Upscaling factor (2, 3, or 4)
            patch_size: Size of sub-images to extract (default: 33)
            patches_per_image: Number of random patches per image
        """
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        
        # Get all image files - that's it! No extraction!
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        print(f"Dataset initialized with {len(self.img_files)} images")
        print(f"Will generate {patches_per_image} random patches per image on-the-fly")
        print(f"Total virtual patches: {len(self.img_files) * patches_per_image}")
    
    def __len__(self):
        """Total number of patches across all images"""
        return len(self.img_files) * self.patches_per_image
    
    def __getitem__(self, idx):
        """
        Generate a random patch on-the-fly
        idx represents: image_idx * patches_per_image + patch_idx
        """
        # Determine which image this patch belongs to
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Load the image
        img_file = self.img_files[img_idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0]
        
        h, w = y_channel.shape
        
        # 🔥 Generate random patch position
        # Use patch_idx as seed for reproducibility within epoch
        # But different epochs will have different random states
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        
        # Extract patch
        hr_patch = y_channel[i:i+self.patch_size, j:j+self.patch_size]
        lr_patch = self._generate_lr_patch(hr_patch)
        
        # Normalize and convert to tensors
        lr_patch = lr_patch.astype(np.float32) / 255.0
        hr_patch = hr_patch.astype(np.float32) / 255.0
        
        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        
        return lr_patch, hr_patch
    
    def _generate_lr_patch(self, hr_patch):
        """Generate low-resolution patch"""
        h, w = hr_patch.shape
        lr_size = (w // self.scale_factor, h // self.scale_factor)
        lr_patch = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_CUBIC)
        lr_patch = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
        return lr_patch


class EpochImageSampler(Sampler):
    """
    Sampler that selects random images each epoch
    Only generates indices for selected images' patches
    """
    def __init__(self, dataset, images_per_epoch, seed=0):
        """
        Args:
            dataset: SRDataset instance
            images_per_epoch: Number of images to sample per epoch
            seed: Random seed (should be different each epoch)
        """
        self.dataset = dataset
        self.images_per_epoch = images_per_epoch
        self.seed = seed
        
        # Calculate which patches to use
        self.indices = self._generate_indices()
    
    def _generate_indices(self):
        """Generate patch indices for randomly selected images"""
        random.seed(self.seed)
        
        # Randomly select images
        num_images = len(self.dataset.img_files)
        selected_imgs = random.sample(range(num_images), 
                                     min(self.images_per_epoch, num_images))
        
        # Generate indices for all patches from selected images
        indices = []
        for img_idx in selected_imgs:
            start_idx = img_idx * self.dataset.patches_per_image
            end_idx = start_idx + self.dataset.patches_per_image
            indices.extend(range(start_idx, end_idx))
        
        return indices
    
    def __iter__(self):
        # Shuffle the indices
        random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)