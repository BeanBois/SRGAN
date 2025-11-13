import cv2 
import torch 
import numpy as np 

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