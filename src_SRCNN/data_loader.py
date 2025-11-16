import torch 
from torch.utils.data import Dataset, SubsetRandomSampler, Sampler
import torchvision.transforms as T

from PIL import Image
import cv2
import numpy as np

import os 
import random 


class EpochImageSampler:
    """
    Samples a random subset of images each epoch, then uses all patches from those images
    """
    def __init__(self, dataset, images_per_epoch=91):
        self.dataset = dataset
        self.images_per_epoch = images_per_epoch
        self.current_epoch = 0
        
    def sample_epoch(self, epoch):
        """Sample random images for this epoch and return their patch indices"""
        self.current_epoch = epoch
        
        # Get random image indices
        random.seed(epoch)  # Different seed per epoch for different images
        sampled_img_indices = random.sample(range(len(self.dataset.img_files)), 
                                           self.images_per_epoch)
        
        # Get all patch indices belonging to these images
        patch_indices = []
        for img_idx in sampled_img_indices:
            start_idx = self.dataset.img_patch_starts[img_idx]
            end_idx = self.dataset.img_patch_starts[img_idx + 1]
            patch_indices.extend(range(start_idx, end_idx))
        
        return patch_indices


class SRDataset(Dataset):
    def __init__(self, img_dir, scale_factor=3, patch_size=33, stride=14, 
                 use_cache=True, crop_size=None):
        """
        SRCNN Dataset that supports per-epoch image sampling
        """
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        self.crop_size = crop_size
        
        # Get all image files
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        print(f"Total images available: {len(self.img_files)}")
        
        # Cache setup
        cache_dir = os.path.join(img_dir, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        crop_str = f"_crop{crop_size}" if crop_size else ""
        cache_filename = f'patches_x{scale_factor}_p{patch_size}_s{stride}{crop_str}_allimgs.pkl'
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        # Load or extract ALL patches from ALL images
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loading from cache: {self.cache_path}")
            import pickle
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.lr_patches = cache_data['lr_patches']
                self.hr_patches = cache_data['hr_patches']
                self.img_patch_starts = cache_data['img_patch_starts']
            print(f"✓ Loaded {len(self.hr_patches)} patches from {len(self.img_files)} images")
        else:
            self._extract_all_patches()
            
            if use_cache:
                print(f"Saving to cache: {self.cache_path}")
                import pickle
                with open(self.cache_path, 'wb') as f:
                    pickle.dump({
                        'lr_patches': self.lr_patches,
                        'hr_patches': self.hr_patches,
                        'img_patch_starts': self.img_patch_starts
                    }, f)
                print("✓ Cache saved")
    
    def _extract_all_patches(self):
        """Extract patches from ALL images and track which patches belong to which image"""
        self.lr_patches = []
        self.hr_patches = []
        self.img_patch_starts = [0]  # Track where each image's patches start
        
        print(f"Extracting patches from all {len(self.img_files)} images...")
        print(f"  patch_size={self.patch_size}, stride={self.stride}")
        if self.crop_size:
            print(f"  crop_size={self.crop_size}")
        
        for idx, img_file in enumerate(self.img_files):
            patches_before = len(self.hr_patches)
            self._extract_patches_from_image(img_file)
            
            # Record the start index for next image
            self.img_patch_starts.append(len(self.hr_patches))
            
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(self.img_files)} images, "
                      f"{len(self.hr_patches)} total patches")
        
        print(f"✓ Total: {len(self.hr_patches)} patches from {len(self.img_files)} images")
        print(f"✓ Average patches per image: {len(self.hr_patches) // len(self.img_files)}")
    
    def _extract_patches_from_image(self, img_file):
        """Extract patches from a single image"""
        img_path = os.path.join(self.img_dir, img_file)
        
        img_pil = Image.open(img_path).convert('RGB')
        
        # Center crop if specified
        if self.crop_size:
            w, h = img_pil.size
            left = max(0, (w - self.crop_size) // 2)
            top = max(0, (h - self.crop_size) // 2)
            right = min(w, left + self.crop_size)
            bottom = min(h, top + self.crop_size)
            img_pil = img_pil.crop((left, top, right, bottom))
        
        # Convert to Y channel
        img_np = np.array(img_pil)
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0]
        
        # Extract all patches with given stride
        h, w = y_channel.shape
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                hr_patch = y_channel[i:i+self.patch_size, j:j+self.patch_size]
                lr_patch = self._generate_lr_patch(hr_patch)
                
                self.hr_patches.append(hr_patch)
                self.lr_patches.append(lr_patch)
    
    def _generate_lr_patch(self, hr_patch):
        """Generate low-resolution patch from high-resolution patch"""
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
# import torch 
# from torch.utils.data import Dataset, SubsetRandomSampler
# import os
# from PIL import Image
# import numpy as np
# import torchvision.transforms as T
# import cv2
# import random 

# def get_epoch_sampler(dataset, images_per_epoch=91, epoch=0):
#     """
#     Create a sampler that randomly selects patches from N random images
    
#     Args:
#         dataset: SRDataset instance
#         images_per_epoch: Number of images to sample (default: 91 like paper)
#         epoch: Current epoch number (for reproducibility)
    
#     Returns:
#         SubsetRandomSampler with indices from randomly selected images
#     """
#     # Seed with epoch for different images each epoch
#     random.seed(epoch)
    
#     # Randomly select image indices
#     num_images = len(dataset.img_files)
#     selected_img_indices = random.sample(range(num_images), 
#                                         min(images_per_epoch, num_images))
    
#     # Get all patch indices from selected images
#     patch_indices = []
#     for img_idx in selected_img_indices:
#         start_idx = dataset.img_patch_starts[img_idx]
#         end_idx = dataset.img_patch_starts[img_idx + 1]
#         patch_indices.extend(range(start_idx, end_idx))
    
#     print(f"Epoch {epoch}: Sampled {len(selected_img_indices)} images, "
#           f"{len(patch_indices)} patches")
    
#     return SubsetRandomSampler(patch_indices)

# class SRDataset(Dataset):
#     def __init__(self, img_dir, scale_factor=3, patch_size=33, stride=14):
#         """
#         Memory-efficient SRCNN Dataset - extracts patches on-the-fly
#         """
#         self.img_dir = img_dir
#         self.scale_factor = scale_factor
#         self.patch_size = patch_size
#         self.stride = stride
        
#         # Get list of image files
#         self.img_files = sorted([f for f in os.listdir(img_dir) 
#                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
#         # Calculate patch indices for each image
#         self.patch_info = []  # [(img_idx, i, j), ...]
        
#         print(f"Calculating patch locations from {len(self.img_files)} images...")
#         for img_idx, img_file in enumerate(self.img_files):
#             img_path = os.path.join(self.img_dir, img_file)
#             img_pil = Image.open(img_path).convert('RGB')
#             h, w = img_pil.size[1], img_pil.size[0]
            
#             # Calculate valid patch positions
#             for i in range(0, h - self.patch_size + 1, self.stride):
#                 for j in range(0, w - self.patch_size + 1, self.stride):
#                     self.patch_info.append((img_idx, i, j))
        
#         print(f"Total patches: {len(self.patch_info)}")
    
#     def __len__(self):
#         return len(self.patch_info)
    
#     def __getitem__(self, idx):
#         """Extract and return a single patch pair on-the-fly"""
#         img_idx, i, j = self.patch_info[idx]
#         img_file = self.img_files[img_idx]
#         img_path = os.path.join(self.img_dir, img_file)
        
#         # Read image and convert to YCbCr
#         img_pil = Image.open(img_path).convert('RGB')
#         img_np = np.array(img_pil)
#         img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        
#         # Extract Y channel patch
#         y_channel = img_ycrcb[:, :, 0]
#         hr_patch = y_channel[i:i+self.patch_size, j:j+self.patch_size]
        
#         # Generate LR patch
#         lr_patch = self._generate_lr_patch(hr_patch)
        
#         # Normalize and convert to tensors
#         lr_patch = lr_patch.astype(np.float32) / 255.0
#         hr_patch = hr_patch.astype(np.float32) / 255.0
        
#         lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
#         hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        
#         return lr_patch, hr_patch
    
#     def _generate_lr_patch(self, hr_patch):
#         """Generate low-resolution patch from high-resolution patch"""
#         h, w = hr_patch.shape
#         lr_size = (w // self.scale_factor, h // self.scale_factor)
#         lr_patch = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_CUBIC)
#         lr_patch = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
#         return lr_patch