import torch 
from torch.utils.data import Dataset, SubsetRandomSampler
import os
from PIL import Image
import numpy as np
import cv2
import random

def get_epoch_sampler(dataset, images_per_epoch=30, epoch=0):
    """
    Create a sampler that randomly selects patches from N random images
    
    Args:
        dataset: SRDataset instance
        images_per_epoch: Number of images to sample
        epoch: Current epoch number (for reproducibility)
    
    Returns:
        SubsetRandomSampler with indices from randomly selected images
    """
    # Resample patches with different random seed each epoch
    # This gives you DIFFERENT patches every epoch!
    random.seed(epoch)
    np.random.seed(epoch)
    
    # Randomly select image indices
    num_images = len(dataset.img_files)
    selected_img_indices = random.sample(range(num_images), 
                                        min(images_per_epoch, num_images))
    
    # Get all patch indices from selected images
    patch_indices = []
    for img_idx in selected_img_indices:
        start_idx = dataset.img_patch_starts[img_idx]
        end_idx = dataset.img_patch_starts[img_idx + 1]
        patch_indices.extend(range(start_idx, end_idx))
    
    print(f"Epoch {epoch}: Sampled {len(selected_img_indices)} images, "
          f"{len(patch_indices)} patches")
    
    return SubsetRandomSampler(patch_indices)

class SRDataset(Dataset):
    def __init__(self, img_dir, scale_factor=3, patch_size=33, 
                 patches_per_image=100,  # 🔥 NEW: Random sampling
                 use_cache=False):  # 🔥 Don't cache random samples
        """
        SRCNN Dataset with random patch sampling per image
        
        Args:
            img_dir: Directory containing training images
            scale_factor: Upscaling factor (2, 3, or 4)
            patch_size: Size of sub-images to extract (default: 33)
            patches_per_image: Number of random patches to sample per image
            use_cache: Don't use cache for random sampling (generates new patches each epoch)
        """
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        
        # Get all image files
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        print(f"Total images available: {len(self.img_files)}")
        print(f"Random sampling {patches_per_image} patches per image")
        
        # Track which patches belong to which image for epoch sampling
        self.img_patch_count = patches_per_image
        
        # Don't pre-extract - will do on-the-fly with random sampling
        self.lr_patches = []
        self.hr_patches = []
        self.img_patch_starts = []
        
        # Extract patches with random sampling
        self._extract_all_patches()
    
    def _extract_all_patches(self):
        """Extract random patches from ALL images"""
        self.lr_patches = []
        self.hr_patches = []
        self.img_patch_starts = [0]
        
        print(f"Extracting {self.patches_per_image} random patches from {len(self.img_files)} images...")
        
        for idx, img_file in enumerate(self.img_files):
            self._extract_random_patches_from_image(img_file)
            self.img_patch_starts.append(len(self.hr_patches))
            
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(self.img_files)} images, "
                      f"{len(self.hr_patches)} total patches")
        
        print(f"✓ Total: {len(self.hr_patches)} patches from {len(self.img_files)} images")
        print(f"✓ Exactly {len(self.hr_patches) // len(self.img_files)} patches per image")
    
    def _extract_random_patches_from_image(self, img_file):
        """
        Extract random patches from a single image
        Instead of grid sampling, randomly pick patch locations
        """
        img_path = os.path.join(self.img_dir, img_file)
        
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0]
        
        h, w = y_channel.shape
        
        # 🔥 Randomly sample patch top-left positions
        for _ in range(self.patches_per_image):
            # Random top-left corner
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            
            # Extract patch
            hr_patch = y_channel[i:i+self.patch_size, j:j+self.patch_size]
            lr_patch = self._generate_lr_patch(hr_patch)
            
            self.hr_patches.append(hr_patch)
            self.lr_patches.append(lr_patch)
    
    def _generate_lr_patch(self, hr_patch):
        """Generate low-resolution patch"""
        h, w = hr_patch.shape
        lr_size = (w // self.scale_factor, h // self.scale_factor)
        lr_patch = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_CUBIC)
        lr_patch = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
        return lr_patch
    
    def __len__(self):
        return len(self.hr_patches)
    
    def __getitem__(self, idx):
        """Return a single patch pair"""
        lr_patch = self.lr_patches[idx].astype(np.float32) / 255.0
        hr_patch = self.hr_patches[idx].astype(np.float32) / 255.0
        
        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        
        return lr_patch, hr_patch
    
    def resample_patches(self):
        """
        Resample all patches (call this at the start of each epoch for variation)
        """
        print(f"Resampling patches for new epoch...")
        self._extract_all_patches()