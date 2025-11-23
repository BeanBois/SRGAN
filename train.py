import torch 
import torch.nn as nn 
import os 
import argparse
import torch.optim as optim

def pretrain_SRResNet(generator, dataloader, num_iterations=1e6):
    """Pre-train generator with MSE loss before GAN training."""
    if os.name == 'nt':
        import torch_directml
        dml = torch_directml.device()
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Phase 1: Pre-training SRResNet with MSE loss...")
    print(f"Training on device: {device}")

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()
    
    generator.to(device)
    generator.train()
    
    iteration = 0
    while iteration < num_iterations:
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            optimizer.zero_grad()
            sr_imgs = generator(lr_imgs)
            loss = mse_loss(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            iteration += 1
            if iteration % 1000 == 0:
                print(f'Pre-train iter {iteration}, MSE: {loss.item():.6f}')
            if iteration >= num_iterations:
                break
    
    return generator


def train_SRGAN(generator, discriminator, dataloader, num_epochs=100, save_interval=10):
    from src_SRGAN import VGGLoss
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
        import torch_directml
        dml = torch_directml.device()
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Phase 2: Training SRGAN with perceptual loss...")
    print(f"Training on device: {device}")
    
    generator.to(device)
    discriminator.to(device)
    content_loss.to(device)  # Move VGG to device too
    
    # Training mode
    generator.train()
    discriminator.train()

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=num_epochs//2, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=num_epochs//2, gamma=0.1)
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
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Epoch statistics
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        print(f'\n==> Epoch [{epoch+1}/{num_epochs}] '
              f'Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}\n')
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint_SRGAN(generator, discriminator, optimizer_G, optimizer_D, 
                          epoch, f'checkpoint_epoch_{epoch+1}.pth')

def save_checkpoint_SRGAN(generator, discriminator, optimizer_G, optimizer_D, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    os.makedirs('model_checkpoints', exist_ok=True)
    filename = os.path.join('model_checkpoints', 'SRGAN' , filename)
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved: {filename}')


def train_SRCNN(model, dataset, val_loader, num_epochs=100, save_interval = 10, images_per_epoch=91, batch_size = 128):
    from src_SRCNN import validate_srcnn, EpochImageSampler
    # see if its Windows or Linux
    if os.name == 'nt':
        import torch_directml
        dml = torch_directml.device()
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f'Using {images_per_epoch} images per epoch')

    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()
    
    # Optimizer: SGD with momentum
    # Different learning rates for different layers (as per paper)
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.conv2.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-5}  # Smaller lr for last layer
    ], momentum=0.9)
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(weights_init)
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()

        epoch_sampler = EpochImageSampler(dataset, images_per_epoch, seed=epoch)
        
        # 🔥 CREATE NEW DATALOADER WITH THIS EPOCH'S SAMPLER
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=epoch_sampler,  # Use sampler instead of shuffle
            num_workers=4,
            pin_memory=True
        )

        epoch_loss = 0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            
            # Compute loss
            loss = criterion(outputs, hr_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}')
        
        # Validate every few epochs
        if (epoch + 1) % 5 == 0:
            validate_srcnn(model, val_loader, device)
        # save model every few epochs 

        if (epoch + 1) % save_interval == 0:
            filename = f'checkpoint_epoch_{epoch+1}.pth'
            filename = os.path.join('model_checkpoints', 'SRCNN' , filename)
            torch.save(model.state_dict(), filename)
    
    return model


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Script to allow choosing of model type")
    parser.add_argument("--model", help="the model you want to train")

    args = parser.parse_args()
    if args.model == 'SRCNN':
        print('Training SRCNN')

        from src_SRCNN import SRDataset, SRCNN

        num_channels = 1  # Y channel only
        f1, f2, f3 = 9, 5, 5
        n1, n2 = 64, 32
        scale_factor = 3
        patch_size = 33
        batch_size = 128
        num_epochs = 100
        images_per_epoch = 91
        patches_per_image = 100

        train_dataset = SRDataset('data/train', scale_factor, patch_size, patches_per_image)
        val_dataset = SRDataset('data/valid', scale_factor, patch_size, patches_per_image)

        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Initialize model
        model = SRCNN(num_channels, f1, f2, f3, n1, n2)
        
        # Train
        model = train_SRCNN(model, train_dataset, val_loader, num_epochs, images_per_epoch=91, batch_size=batch_size)
        
    else:
        print('Training SRGAN')

        from src_SRGAN import GenerativeNetwork, DiscriminatoryNetwork, ImgDataset
        
        generator = GenerativeNetwork()
        discriminator = DiscriminatoryNetwork()
        
        training_dataset = ImgDataset('data/train', downscale_factor=4)
        
        dataloader = DataLoader(
            training_dataset, 
            batch_size=32,          
            shuffle=True,           
            num_workers=4,          
            pin_memory=True         
        )
        
        print(f"Dataset size: {len(training_dataset)}")
        print(f"Number of batches: {len(dataloader)}")
        print(f"Batch size: {dataloader.batch_size}")
        
        generator = pretrain_SRResNet(
            generator, 
            dataloader, 
            num_iterations=20000, # 10^6 // (800//16)
        )
        
        
        train_SRGAN(generator, discriminator, dataloader, num_epochs=8000,save_interval=400) # 200000 // (800//16)