import cv2 
import torch 
import numpy as np 
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_psnr(img1, img2, max_pixel=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W]
        img2: Second image tensor [B, C, H, W] or [C, H, W]
        max_pixel: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_pixel=1.0):
    """
    Calculate Structural Similarity Index between two images.
    Simplified version for grayscale or per-channel calculation.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        window_size: Size of the sliding window
        max_pixel: Maximum pixel value
    
    Returns:
        SSIM value
    """
    C1 = (0.01 * max_pixel) ** 2
    C2 = (0.03 * max_pixel) ** 2
    
    # Simple average pooling as approximation
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def denormalize_image(tensor, from_range='[-1,1]'):
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        from_range: Input range, either '[-1,1]' or '[0,1]'
    
    Returns:
        Denormalized tensor in [0, 1]
    """
    if from_range == '[-1,1]':
        return (tensor + 1.0) / 2.0
    return tensor


def save_comparison_images(lr_img, sr_img, hr_img, save_path, img_idx):
    """
    Save side-by-side comparison of LR, SR, and HR images.
    
    Args:
        lr_img: Low-resolution image tensor [C, H, W]
        sr_img: Super-resolved image tensor [C, H, W]
        hr_img: High-resolution image tensor [C, H, W]
        save_path: Directory to save images
        img_idx: Image index for naming
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Denormalize images
    lr_img = denormalize_image(lr_img, '[0,1]').cpu().numpy().transpose(1, 2, 0)
    sr_img = denormalize_image(sr_img, '[-1,1]').cpu().numpy().transpose(1, 2, 0)
    hr_img = denormalize_image(hr_img, '[-1,1]').cpu().numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1]
    lr_img = np.clip(lr_img, 0, 1)
    sr_img = np.clip(sr_img, 0, 1)
    hr_img = np.clip(hr_img, 0, 1)
    
    # Resize LR to match HR size for visualization
    lr_img_pil = Image.fromarray((lr_img * 255).astype('uint8'))
    lr_img_upscaled = lr_img_pil.resize((hr_img.shape[1], hr_img.shape[0]), Image.BICUBIC)
    lr_img = np.array(lr_img_upscaled) / 255.0
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(lr_img)
    axes[0].set_title('LR (Bicubic Upscaled)')
    axes[0].axis('off')
    
    axes[1].imshow(sr_img)
    axes[1].set_title('SR (SRGAN)')
    axes[1].axis('off')
    
    axes[2].imshow(hr_img)
    axes[2].set_title('HR (Ground Truth)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'comparison_{img_idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual images
    Image.fromarray((sr_img * 255).astype('uint8')).save(
        os.path.join(save_path, f'sr_{img_idx:04d}.png'))
    Image.fromarray((hr_img * 255).astype('uint8')).save(
        os.path.join(save_path, f'hr_{img_idx:04d}.png'))


def evaluate_SRGAN(generator, val_loader, device, checkpoint_path, 
                   save_images=True, num_images_to_save=10, output_dir='evaluation_results'):
    """
    Evaluate SRGAN generator on validation set.
    
    Args:
        generator: Generator network (will be loaded with checkpoint)
        val_loader: DataLoader for validation set
        device: Device to run evaluation on
        checkpoint_path: Path to the generator checkpoint (.pth file)
        save_images: Whether to save comparison images
        num_images_to_save: Number of comparison images to save
        output_dir: Directory to save results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle both direct state_dict and full checkpoint formats
    if isinstance(state_dict, dict) and 'generator_state_dict' in state_dict:
        generator.load_state_dict(state_dict['generator_state_dict'])
        print(f"Loaded from epoch: {state_dict.get('epoch', 'unknown')}")
    else:
        generator.load_state_dict(state_dict)
    
    print("Checkpoint loaded successfully!")
    
    # Move model to device and set to eval mode
    generator.to(device)
    generator.eval()
    
    # Metrics storage
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    
    # Create output directory
    if save_images:
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
    
    print("\nEvaluating on validation set...")
    print("="*60)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Handle both 2-tuple and 3-tuple returns (with/without image names)
            if len(batch_data) == 3:
                lr_imgs, hr_imgs, img_names = batch_data
            else:
                lr_imgs, hr_imgs = batch_data
                img_names = None
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate SR images
            sr_imgs = generator(lr_imgs)
            
            # Calculate metrics for each image in batch
            batch_size = lr_imgs.size(0)
            for i in range(batch_size):
                # Denormalize for metric calculation (both should be in [0, 1])
                sr_img_norm = denormalize_image(sr_imgs[i:i+1], '[-1,1]')
                hr_img_norm = denormalize_image(hr_imgs[i:i+1], '[-1,1]')
                
                # Calculate PSNR
                psnr = calculate_psnr(sr_img_norm, hr_img_norm, max_pixel=1.0)
                psnr_scores.append(psnr)
                
                # Calculate SSIM
                ssim = calculate_ssim(sr_img_norm, hr_img_norm, max_pixel=1.0)
                ssim_scores.append(ssim)
                
                # Calculate MSE
                mse = torch.mean((sr_img_norm - hr_img_norm) ** 2).item()
                mse_scores.append(mse)
                
                # Save comparison images
                if save_images and (batch_idx * batch_size + i) < num_images_to_save:
                    img_idx = batch_idx * batch_size + i
                    save_comparison_images(
                        lr_imgs[i], sr_imgs[i], hr_imgs[i],
                        images_dir, img_idx
                    )
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)
    std_mse = np.std(mse_scores)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of images evaluated: {len(psnr_scores)}")
    print(f"\nAverage PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Average MSE:  {avg_mse:.6f} ± {std_mse:.6f}")
    print("="*60)
    
    # Save results to file
    results_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("SRGAN EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Number of images: {len(psnr_scores)}\n\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"Average MSE:  {avg_mse:.6f} ± {std_mse:.6f}\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    if save_images:
        print(f"Comparison images saved to: {images_dir}")
    
    # Return metrics dictionary
    results = {
        'avg_psnr': avg_psnr,
        'std_psnr': std_psnr,
        'avg_ssim': avg_ssim,
        'std_ssim': std_ssim,
        'avg_mse': avg_mse,
        'std_mse': std_mse,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores,
        'mse_scores': mse_scores
    }
    
    return results

# SRCNN evaluation function
def eval_SRCNN(model, image_path, scale_factor=3, device='cuda'):
    """
    Super-resolve a single image
    """
    model.eval()
    
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Extract Y channel
    y_channel = img_ycrcb[:, :, 0]
    
    # Downsample and upsample (bicubic interpolation)
    h, w = y_channel.shape
    lr_size = (w // scale_factor, h // scale_factor)
    y_lr = cv2.resize(y_channel, lr_size, interpolation=cv2.INTER_CUBIC)
    y_bicubic = cv2.resize(y_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize and convert to tensor
    y_input = y_bicubic.astype(np.float32) / 255.0
    y_input = torch.from_numpy(y_input).unsqueeze(0).unsqueeze(0).to(device)
    
    # Super-resolve
    with torch.no_grad():
        y_sr = model(y_input)
    
    # Convert back to numpy
    y_sr = y_sr.squeeze().cpu().numpy()
    y_sr = np.clip(y_sr * 255.0, 0, 255).astype(np.uint8)
    
    # Reconstruct RGB image
    img_ycrcb[:, :, 0] = y_sr
    img_sr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return img_sr, y_bicubic, y_sr
# SRCNN evaluation function
def eval_SRCNN(model, image_path, scale_factor=3, device='cuda'):
    """
    Super-resolve a single image
    """
    model.eval()
    
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Extract Y channel
    y_channel = img_ycrcb[:, :, 0]
    
    # Downsample and upsample (bicubic interpolation)
    h, w = y_channel.shape
    lr_size = (w // scale_factor, h // scale_factor)
    y_lr = cv2.resize(y_channel, lr_size, interpolation=cv2.INTER_CUBIC)
    y_bicubic = cv2.resize(y_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize and convert to tensor
    y_input = y_bicubic.astype(np.float32) / 255.0
    y_input = torch.from_numpy(y_input).unsqueeze(0).unsqueeze(0).to(device)
    
    # Super-resolve
    with torch.no_grad():
        y_sr = model(y_input)
    
    # Convert back to numpy
    y_sr = y_sr.squeeze().cpu().numpy()
    y_sr = np.clip(y_sr * 255.0, 0, 255).astype(np.uint8)
    
    # Reconstruct RGB image
    img_ycrcb[:, :, 0] = y_sr
    img_sr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return img_sr, y_bicubic, y_sr

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Script to allow choosing of model type")
    parser.add_argument("--model", help="the model you want to train", default="SRGAN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to generator checkpoint (.pth file)")
    parser.add_argument("--val_dir", type=str, default='data/valid',
                        help="Path to validation dataset directory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of comparison images to save")
    parser.add_argument("--output_dir", type=str, default='evaluation_results',
                        help="Directory to save evaluation results")
    parser.add_argument("--hr_size", type=int, default=1000,
                        help="High-resolution image size")
    parser.add_argument("--downscale_factor", type=int, default=4,
                        help="Downscale factor")
    args = parser.parse_args()
    if args.model == 'SRCNN':
        from src_SRCNN import SRCNN
        model = SRCNN()
        model.load_state_dict(torch.load('model_checkpoints/SRCNN/checkpoint.pth'))
    else:
        from src_SRGAN import GenerativeNetwork, WholeImageDataset
        if os.name == 'nt':
                try:
                    import torch_directml
                    device = torch_directml.device()
                    print("Using DirectML device")
                except ImportError:
                    device = torch.device('cpu')
                    print("DirectML not available, using CPU")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
        generator = GenerativeNetwork()
        
        # Create validation dataset and dataloader
        val_dataset = WholeImageDataset(
            img_dir=args.val_dir,
            downscale_factor=args.downscale_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid issues with variable image sizes
            pin_memory=False
        )
        
        print(f"\nValidation set size: {len(val_dataset)} images")
        print(f"Number of batches: {len(val_loader)}")
        
        # Run evaluation
        results = evaluate_SRGAN(
            generator=generator,
            val_loader=val_loader,
            device=device,
            checkpoint_path=args.checkpoint,
            save_images=True,
            num_images_to_save=args.num_images,
            output_dir=args.output_dir
        )
        
        print("\nEvaluation complete!")