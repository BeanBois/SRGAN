import torch 
import torch.nn as nn 
import os 
import argparse
import torch.optim as optim

def pretrain_SRResNet(generator, dataloader, num_iterations=1_000_000, save_interval=10000):
    """
    Pre-train generator with MSE loss before GAN training.
    Paper: 10^6 iterations with lr=1e-4
    """
    if os.name == 'nt':
        import torch_directml
        dml = torch_directml.device()
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Phase 1: Pre-training SRResNet with MSE loss")
    print("="*60)
    print(f"Training on device: {device}")
    print(f"Target iterations: {int(num_iterations):,}")
    print(f"Images per epoch: {len(dataloader.dataset):,}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Iterations per epoch: {len(dataloader):,}")
    print(f"Expected epochs: {int(num_iterations / len(dataloader))}")

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()
    
    generator.to(device)
    generator.train()
    
    iteration = 0
    epoch = 0
    
    while iteration < num_iterations:
        epoch += 1
        epoch_loss = 0.0
        num_batches = 0
        
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            optimizer.zero_grad()
            sr_imgs = generator(lr_imgs)
            
            # MSE loss on [-1, 1] range as per paper
            loss = mse_loss(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            iteration += 1
            epoch_loss += loss.item()
            num_batches += 1
            
            # Check for NaN
            if torch.isnan(loss):
                print(f'⚠️  NaN detected at iteration {iteration}!')
                return None

            # Progress logging
            if iteration % 1000 == 0:
                print(f'[Pre-train] Iter {iteration:,}/{int(num_iterations):,} '
                      f'(Epoch {epoch}) MSE: {loss.item():.6f}')
            
            # Save checkpoints
            if iteration % save_interval == 0:
                os.makedirs('model_checkpoints/SRGAN', exist_ok=True)
                save_path = f'model_checkpoints/SRGAN/SRResNet_pretrain_iter_{iteration}.pth'
                torch.save(generator.state_dict(), save_path)
                print(f'✅ Checkpoint saved: {save_path}')
            
            if iteration >= num_iterations:
                break
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f'==> Epoch {epoch} completed: Avg MSE = {avg_epoch_loss:.6f}')
        
        if iteration >= num_iterations:
            break
    
    # Save final pre-trained model
    os.makedirs('model_checkpoints/SRGAN', exist_ok=True)
    final_save_path = 'model_checkpoints/SRGAN/SRResNet_pretrained_final.pth'
    torch.save(generator.state_dict(), final_save_path)
    print(f'✅ Final pre-trained model saved: {final_save_path}')
    
    return generator

def train_SRGAN(generator, discriminator, dataloader, num_iterations=200_000, 
                lr_phase1=1e-4, lr_phase2=1e-5, phase1_iters=100_000, save_interval=10000):
    """
    Train SRGAN with perceptual loss.
    Paper: 
    - 10^5 iterations at lr=1e-4
    - 10^5 iterations at lr=1e-5
    - Total: 2×10^5 iterations
    """
    from src import VGGLoss
    
    if os.name == 'nt':
        import torch_directml
        dml = torch_directml.device()
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Phase 2: Training SRGAN with perceptual loss")
    print("="*60)
    print(f"Training on device: {device}")
    print(f"Total iterations: {num_iterations:,}")
    print(f"Phase 1: {phase1_iters:,} iters at lr={lr_phase1}")
    print(f"Phase 2: {num_iterations - phase1_iters:,} iters at lr={lr_phase2}")
    print(f"Iterations per epoch: {len(dataloader):,}")
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_phase1, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_phase1, betas=(0.9, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = VGGLoss(feature_layer=36)  # VGG5.4 = layer 36
    
    generator.to(device)
    discriminator.to(device)
    content_loss.to(device)
    
    generator.train()
    discriminator.train()
    
    iteration = 0
    epoch = 0
    
    while iteration < num_iterations:
        epoch += 1
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for lr_imgs, hr_imgs in dataloader:
            # Move batch to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            batch_size = lr_imgs.size(0)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ===== Train Discriminator =====
            optimizer_D.zero_grad()
            
            # Real images
            real_output = discriminator(hr_imgs)
            d_loss_real = adversarial_loss(real_output, real_labels)
            
            # Fake images
            sr_imgs = generator(lr_imgs)
            fake_output = discriminator(sr_imgs.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # ===== Train Generator =====
            optimizer_G.zero_grad()
            
            sr_imgs = generator(lr_imgs)
            
            # Adversarial loss (weighted by 1e-3 as per paper)
            gen_output = discriminator(sr_imgs)
            adversarial_g_loss = adversarial_loss(gen_output, real_labels)
            
            # Content loss (VGG perceptual loss)
            perceptual_loss = content_loss(sr_imgs, hr_imgs)
            
            # Total generator loss (Equation 3 in paper)
            g_loss = perceptual_loss + 1e-3 * adversarial_g_loss
            g_loss.backward()
            optimizer_G.step()
            
            iteration += 1
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1
            
            # Learning rate schedule: switch to lower lr after phase1_iters
            if iteration == phase1_iters:
                print(f"\n🔄 Switching learning rate: {lr_phase1} -> {lr_phase2}\n")
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = lr_phase2
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = lr_phase2
            
            # Progress logging
            if iteration % 1000 == 0:
                print(f'[SRGAN] Iter {iteration:,}/{num_iterations:,} '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} '
                      f'Percep: {perceptual_loss.item():.4f} Adv: {adversarial_g_loss.item():.4f}')
            
            # Save checkpoints
            if iteration % save_interval == 0:
                save_checkpoint_SRGAN(
                    generator, discriminator, optimizer_G, optimizer_D,
                    iteration, f'checkpoint_iter_{iteration}.pth'
                )
            
            if iteration >= num_iterations:
                break
        
        # Epoch summary
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f'==> Epoch {epoch}: Avg D_loss={avg_d_loss:.4f}, Avg G_loss={avg_g_loss:.4f}')
        
        if iteration >= num_iterations:
            break
    
    # Save final model
    save_checkpoint_SRGAN(
        generator, discriminator, optimizer_G, optimizer_D,
        iteration, 'SRGAN_final.pth'
    )
    print(f'✅ Final SRGAN model saved')

def save_checkpoint_SRGAN(generator, discriminator, optimizer_G, optimizer_D, iteration, filename):
    checkpoint = {
        'iteration': iteration,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    os.makedirs('model_checkpoints/SRGAN', exist_ok=True)
    filepath = os.path.join('model_checkpoints', 'SRGAN', filename)
    torch.save(checkpoint, filepath)
    print(f'💾 Checkpoint saved: {filepath}')


def train_SRCNN(model, dataset, val_loader, num_epochs=100, save_interval=10, 
                images_per_epoch=91, batch_size=128):
    from src_SRCNN import validate_srcnn, EpochImageSampler
    from torch.utils.data import DataLoader
    
    if os.name == 'nt':
        import torch_directml
        dml = torch_directml.device()
        device = dml
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    print(f'Using {images_per_epoch} images per epoch')

    criterion = nn.MSELoss()
    
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.conv2.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-5}
    ], momentum=0.9)
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(weights_init)
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_sampler = EpochImageSampler(dataset, images_per_epoch, seed=epoch)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=epoch_sampler,
            num_workers=4,
            pin_memory=True
        )

        epoch_loss = 0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}')
        
        if (epoch + 1) % 5 == 0:
            validate_srcnn(model, val_loader, device)
        
        if (epoch + 1) % save_interval == 0:
            os.makedirs('model_checkpoints/SRCNN', exist_ok=True)
            filename = f'checkpoint_epoch_{epoch+1}.pth'
            filepath = os.path.join('model_checkpoints', 'SRCNN', filename)
            torch.save(model.state_dict(), filepath)
    
    return model


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src import GenerativeNetwork, DiscriminatoryNetwork
    from src.data_loader import ImgDataset  
    
    parser = argparse.ArgumentParser(description="Train SRGAN following the paper")
    parser.add_argument("--pretrained_model", help="Path to pretrained SRResNet model", default=None)
    parser.add_argument("--data_dir", help="Training data directory", default='data/train')
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size (paper uses 16)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    print('='*60)
    print('Training SRGAN following paper specifications')
    print('='*60)
    
    # Initialize networks
    generator = GenerativeNetwork()
    discriminator = DiscriminatoryNetwork()
    
    # Create dataset (ONE patch per image, different images per batch)
    training_dataset = ImgDataset(
        args.data_dir,
        hr_size=96,
        downscale_factor=4,
        is_training=True
    )
    
    # DataLoader with batch_size=16 as per paper
    dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle to get different images each batch
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    print(f"\nDataset Configuration:")
    print(f"  Total images: {len(training_dataset):,}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Batches per epoch: {len(dataloader):,}")
    print(f"  Patches per batch: {args.batch_size} (from {args.batch_size} DISTINCT images)")
    
    # Phase 1: Pre-train SRResNet with MSE (10^6 iterations)
    if args.pretrained_model is None:
        print("\n" + "="*60)
        print("Starting Phase 1: Pre-training SRResNet")
        print("="*60)
        generator = pretrain_SRResNet(
            generator,
            dataloader,
            num_iterations=1_000_000,  # 10^6 as per paper
            save_interval=10_000
        )
        
        if generator is None:
            print("❌ Pre-training failed due to NaN. Exiting.")
            exit(1)
    else:
        print(f"\n✅ Loading pre-trained model: {args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model)
        generator.load_state_dict(state_dict)
    
    # Phase 2: Train SRGAN with perceptual loss (2×10^5 iterations)
    print("\n" + "="*60)
    print("Starting Phase 2: Training SRGAN")
    print("="*60)
    
    train_SRGAN(
        generator,
        discriminator,
        dataloader,
        num_iterations=200_000,  # 2×10^5 as per paper
        lr_phase1=1e-4,          # First 10^5 iterations
        lr_phase2=1e-5,          # Second 10^5 iterations
        phase1_iters=100_000,
        save_interval=10_000
    )
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print("="*60)