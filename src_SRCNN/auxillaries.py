import torch 

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def validate_srcnn(model, val_loader, device):
    """
    Validate SRCNN model
    """
    model.eval()
    total_psnr = 0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            outputs = model(lr_imgs)
            
            # Calculate PSNR
            psnr = calculate_psnr(outputs, hr_imgs)
            total_psnr += psnr
    
    avg_psnr = total_psnr / len(val_loader)
    print(f'Validation PSNR: {avg_psnr:.2f} dB')
    
    return avg_psnr