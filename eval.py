import cv2 
import torch 
import numpy as np 

# SRGAN evaluation function
def load_checkpoint_SRGAN(generator, discriminator, optimizer_G, optimizer_D, filename,type):
    checkpoint = torch.load(filename)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    epoch = checkpoint['epoch']
    os.makedirs('model_checkpoints', exist_ok=True)
    filename = os.path.join('model_checkpoints', 'SRGAN', filename)
    print(f'Checkpoint loaded: {filename} (epoch {epoch})')
    return epoch



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

    parser = argparse.ArgumentParser(description="Script to allow choosing of model type")
    parser.add_argument("--model", help="the model you want to train")

    args = parser.parse_args()
    if args.model == 'SRCNN':
        from src_SRCNN import SRCNN
        model = SRCNN()
        model.load_state_dict(torch.load('model_checkpoints/SRCNN/checkpoint.pth'))
    else:
        from src_SRGAN import SRGAN
        
        generator = SRGAN()
        discriminator = SRGAN()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        epoch = load_checkpoint_SRGAN(generator, discriminator, optimizer_G, optimizer_D, 'checkpoint_epoch_100.pth', args.model)

        