import torch 
import torch.nn as nn 
from src_SRGAN import VGGLoss
import torch_directml
import os 
dml = torch_directml.device()



def train_SRGAN(generator, discriminator, dataloader, num_epochs=100, save_interval=10):
    """
    Train SRGAN with proper batch handling.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        dataloader: DataLoader with batched data
        num_epochs: Number of training epochs
        save_interval: Save checkpoint every N epochs
    """
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = VGGLoss(feature_layer=36)  
    
    # see if its Windows or Linux
    if os.name == 'nt':
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    generator.to(device)
    discriminator.to(device)
    content_loss.to(device)  # Move VGG to device too
    
    # Training mode
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            # Move batch to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Get actual batch size (last batch might be smaller)
            batch_size = lr_imgs.size(0)
            
            # Real and fake labels for this batch
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            #  Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images - discriminator should output ~1
            real_output = discriminator(hr_imgs)
            d_loss_real = adversarial_loss(real_output, real_labels)
            
            # Fake images - discriminator should output ~0
            sr_imgs = generator(lr_imgs)
            fake_output = discriminator(sr_imgs.detach())  # detach to not backprop through generator
            d_loss_fake = adversarial_loss(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            #  Train Generator
            optimizer_G.zero_grad()
            
            # Generate SR images 
            sr_imgs = generator(lr_imgs)
            
            # Adversarial loss 
            gen_output = discriminator(sr_imgs)
            adversarial_g_loss = adversarial_loss(gen_output, real_labels)
            
            # Content loss 
            perceptual_loss = content_loss(sr_imgs, hr_imgs)
            
            # Total generator loss 
            g_loss = perceptual_loss + 1e-3 * adversarial_g_loss
            g_loss.backward()
            optimizer_G.step()
            
            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Print progress
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} '
                      f'Perceptual: {perceptual_loss.item():.4f} Adversarial: {adversarial_g_loss.item():.4f}')
        
        # Epoch statistics
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        print(f'\n==> Epoch [{epoch+1}/{num_epochs}] '
              f'Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}\n')
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, 
                          epoch, f'SRGAN{os.sep}checkpoint_epoch_{epoch+1}.pth')

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    os.makedirs('model_checkpoints', exist_ok=True)
    filename = os.path.join('model_checkpoints', filename)
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved: {filename}')

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, filename,type):
    checkpoint = torch.load(filename)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    epoch = checkpoint['epoch']
    os.makedirs('model_checkpoints', exist_ok=True)
    filename = f'{type}{os.sep}{filename}'
    filename = os.path.join('model_checkpoints', filename)
    print(f'Checkpoint loaded: {filename} (epoch {epoch})')
    return epoch

if __name__ == '__main__':
    from src_SRGAN import GenerativeNetwork, DiscriminatoryNetwork, ImgDataset
    from torch.utils.data import DataLoader
    
    generator = GenerativeNetwork()
    discriminator = DiscriminatoryNetwork()
    
    training_dataset = ImgDataset('data/train', downscale_factor=4)
    
    dataloader = DataLoader(
        training_dataset, 
        batch_size=16,          
        shuffle=True,           
        num_workers=4,          
        pin_memory=True         
    )
    
    print(f"Dataset size: {len(training_dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    
    # Train the model
    train(generator, discriminator, dataloader, num_epochs=100)