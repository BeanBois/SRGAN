import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
      
# Import the components (simplified versions for testing)
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super(UpsampleBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class SimpleGenerator(nn.Module):
    """Simplified generator for testing"""
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(channels=64),
            ResidualBlock(channels=64)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.upsample = UpsampleBlock(64, upscale_factor=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )
        
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.residual_blocks(out1)
        out2 = self.conv2(out)
        out = out1 + out2
        out = self.upsample(out)
        out = self.conv3(out)
        return out

class SimpleDiscriminator(nn.Module):
    """Simplified discriminator for testing"""
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.conv_blocks(x)
        out = self.classifier(out)
        return out

def check_gradients(model, model_name):
    """Check if gradients are computed and valid"""
    print(f"\n{'='*60}")
    print(f"Checking gradients for {model_name}")
    print(f"{'='*60}")
    
    has_gradients = False
    grad_stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_gradients = True
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_min = param.grad.min().item()
                grad_max = param.grad.max().item()
                grad_norm = param.grad.norm().item()
                
                # Check for NaN or Inf
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                
                grad_stats.append({
                    'name': name,
                    'shape': tuple(param.grad.shape),
                    'mean': grad_mean,
                    'std': grad_std,
                    'min': grad_min,
                    'max': grad_max,
                    'norm': grad_norm,
                    'has_nan': has_nan,
                    'has_inf': has_inf
                })
                
                status = "✓ OK"
                if has_nan:
                    status = "✗ HAS NaN"
                elif has_inf:
                    status = "✗ HAS Inf"
                elif grad_norm == 0:
                    status = "⚠ ZERO GRADIENT"
                
                print(f"\n{name}:")
                print(f"  Shape: {param.grad.shape}")
                print(f"  Mean: {grad_mean:.6e}, Std: {grad_std:.6e}")
                print(f"  Min: {grad_min:.6e}, Max: {grad_max:.6e}")
                print(f"  Norm: {grad_norm:.6e}")
                print(f"  Status: {status}")
            else:
                print(f"\n{name}: ✗ NO GRADIENT")
    
    return has_gradients, grad_stats

def test_directml_backprop():
    """Main test function for DirectML backward propagation"""
    
    print("="*60)
    print("DirectML Backward Propagation Test")
    print("="*60)
    
    # Try to use DirectML if available, otherwise use CPU
    try:
        import torch_directml
        dml = torch_directml.device()
        device = dml
        print(f"\n✓ Using DirectML device")
    except:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\n⚠ DirectML not available, using {device}")
        except:
            device = torch.device("cpu")
            print(f"\n⚠ Using CPU fallback")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Test parameters
    batch_size = 2
    img_size = 32  # Small size for quick testing
    lr = 1e-4
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Learning rate: {lr}")
    
    # Create models
    print("\n" + "="*60)
    print("Initializing models...")
    print("="*60)
    
    generator = SimpleGenerator().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    print(f"✓ Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()
    
    print("\n" + "="*60)
    print("Creating test data...")
    print("="*60)
    
    # Create dummy data
    lr_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    hr_images = torch.randn(batch_size, 3, img_size*2, img_size*2).to(device)
    
    print(f"✓ Low-res input: {lr_images.shape}")
    print(f"✓ High-res target: {hr_images.shape}")
    
    # ========================================
    # Test 1: Generator backward pass
    # ========================================
    print("\n" + "="*60)
    print("TEST 1: Generator Backward Pass")
    print("="*60)
    
    gen_optimizer.zero_grad()
    
    print("\nForward pass...")
    fake_images = generator(lr_images)
    print(f"✓ Generated images: {fake_images.shape}")
    
    print("\nComputing losses...")
    # Content loss
    g_content_loss = content_loss(fake_images, hr_images)
    print(f"  Content loss: {g_content_loss.item():.6f}")
    
    # Adversarial loss
    fake_validity = discriminator(fake_images)
    real_labels = torch.ones_like(fake_validity).to(device)
    g_adversarial_loss = adversarial_loss(fake_validity, real_labels)
    print(f"  Adversarial loss: {g_adversarial_loss.item():.6f}")
    
    # Total generator loss
    g_loss = g_content_loss + 1e-3 * g_adversarial_loss
    print(f"  Total generator loss: {g_loss.item():.6f}")
    
    print("\nBackward pass...")
    g_loss.backward()
    
    has_grads, grad_stats = check_gradients(generator, "Generator")
    
    if has_grads:
        print("\n✓ Generator backward pass SUCCESSFUL")
        print(f"  Gradient norm summary: min={min(s['norm'] for s in grad_stats):.6e}, "
              f"max={max(s['norm'] for s in grad_stats):.6e}")
    else:
        print("\n✗ Generator backward pass FAILED - No gradients computed")
    
    print("\nOptimizer step...")
    gen_optimizer.step()
    print("✓ Optimizer step completed")
    
    # ========================================
    # Test 2: Discriminator backward pass
    # ========================================
    print("\n" + "="*60)
    print("TEST 2: Discriminator Backward Pass")
    print("="*60)
    
    disc_optimizer.zero_grad()
    
    print("\nForward pass on real images...")
    real_validity = discriminator(hr_images)
    real_labels = torch.ones_like(real_validity).to(device)
    d_real_loss = adversarial_loss(real_validity, real_labels)
    print(f"  Real loss: {d_real_loss.item():.6f}")
    print(f"  Real validity: mean={real_validity.mean().item():.4f}")
    
    print("\nForward pass on fake images...")
    fake_images = generator(lr_images).detach()  # Detach to avoid generator gradients
    fake_validity = discriminator(fake_images)
    fake_labels = torch.zeros_like(fake_validity).to(device)
    d_fake_loss = adversarial_loss(fake_validity, fake_labels)
    print(f"  Fake loss: {d_fake_loss.item():.6f}")
    print(f"  Fake validity: mean={fake_validity.mean().item():.4f}")
    
    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2
    print(f"  Total discriminator loss: {d_loss.item():.6f}")
    
    print("\nBackward pass...")
    d_loss.backward()
    
    has_grads, grad_stats = check_gradients(discriminator, "Discriminator")
    
    if has_grads:
        print("\n✓ Discriminator backward pass SUCCESSFUL")
        print(f"  Gradient norm summary: min={min(s['norm'] for s in grad_stats):.6e}, "
              f"max={max(s['norm'] for s in grad_stats):.6e}")
    else:
        print("\n✗ Discriminator backward pass FAILED - No gradients computed")
    
    print("\nOptimizer step...")
    disc_optimizer.step()
    print("✓ Optimizer step completed")
    
    # ========================================
    # Test 3: Multi-step training simulation
    # ========================================
    print("\n" + "="*60)
    print("TEST 3: Multi-Step Training Simulation")
    print("="*60)
    
    num_steps = 5
    print(f"\nRunning {num_steps} training steps...")
    
    for step in range(num_steps):
        # Train discriminator
        disc_optimizer.zero_grad()
        fake_images = generator(lr_images).detach()
        real_validity = discriminator(hr_images)
        fake_validity = discriminator(fake_images)
        d_loss = (adversarial_loss(real_validity, torch.ones_like(real_validity).to(device)) + 
                  adversarial_loss(fake_validity, torch.zeros_like(fake_validity).to(device))) / 2
        d_loss.backward()
        disc_optimizer.step()
        
        # Train generator
        gen_optimizer.zero_grad()
        fake_images = generator(lr_images)
        fake_validity = discriminator(fake_images)
        g_loss = (content_loss(fake_images, hr_images) + 
                  1e-3 * adversarial_loss(fake_validity, torch.ones_like(fake_validity).to(device)))
        g_loss.backward()
        gen_optimizer.step()
        
        print(f"  Step {step+1}/{num_steps}: G_loss={g_loss.item():.6f}, D_loss={d_loss.item():.6f}")
    
    print("\n✓ Multi-step training completed successfully")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\n✓ All backward propagation tests completed")
    print(f"✓ Device: {device}")
    print(f"✓ Generator trainable parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad):,}")
    print(f"✓ Discriminator trainable parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}")
    print("\nIf you see gradients computed with reasonable values (not NaN/Inf),")
    print("then DirectML backward propagation is working correctly!")

def validate_with_cpu():
    """Main test function for DirectML backward propagation"""
    
    print("="*60)
    print("USING CPU")
    print("="*60)
    
    # Try to use DirectML if available, otherwise use CPU
    device = torch.device("cpu")

    
    print(f"PyTorch version: {torch.__version__}")
    
    # Test parameters
    batch_size = 2
    img_size = 32  # Small size for quick testing
    lr = 1e-4
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Learning rate: {lr}")
    
    # Create models
    print("\n" + "="*60)
    print("Initializing models...")
    print("="*60)
    
    generator = SimpleGenerator().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    print(f"✓ Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()
    
    print("\n" + "="*60)
    print("Creating test data...")
    print("="*60)
    
    # Create dummy data
    lr_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    hr_images = torch.randn(batch_size, 3, img_size*2, img_size*2).to(device)
    
    print(f"✓ Low-res input: {lr_images.shape}")
    print(f"✓ High-res target: {hr_images.shape}")
    
    # ========================================
    # Test 1: Generator backward pass
    # ========================================
    print("\n" + "="*60)
    print("TEST 1: Generator Backward Pass")
    print("="*60)
    
    gen_optimizer.zero_grad()
    
    print("\nForward pass...")
    fake_images = generator(lr_images)
    print(f"✓ Generated images: {fake_images.shape}")
    
    print("\nComputing losses...")
    # Content loss
    g_content_loss = content_loss(fake_images, hr_images)
    print(f"  Content loss: {g_content_loss.item():.6f}")
    
    # Adversarial loss
    fake_validity = discriminator(fake_images)
    real_labels = torch.ones_like(fake_validity).to(device)
    g_adversarial_loss = adversarial_loss(fake_validity, real_labels)
    print(f"  Adversarial loss: {g_adversarial_loss.item():.6f}")
    
    # Total generator loss
    g_loss = g_content_loss + 1e-3 * g_adversarial_loss
    print(f"  Total generator loss: {g_loss.item():.6f}")
    
    print("\nBackward pass...")
    g_loss.backward()
    
    has_grads, grad_stats = check_gradients(generator, "Generator")
    
    if has_grads:
        print("\n✓ Generator backward pass SUCCESSFUL")
        print(f"  Gradient norm summary: min={min(s['norm'] for s in grad_stats):.6e}, "
              f"max={max(s['norm'] for s in grad_stats):.6e}")
    else:
        print("\n✗ Generator backward pass FAILED - No gradients computed")
    
    print("\nOptimizer step...")
    gen_optimizer.step()
    print("✓ Optimizer step completed")
    
    # ========================================
    # Test 2: Discriminator backward pass
    # ========================================
    print("\n" + "="*60)
    print("TEST 2: Discriminator Backward Pass")
    print("="*60)
    
    disc_optimizer.zero_grad()
    
    print("\nForward pass on real images...")
    real_validity = discriminator(hr_images)
    real_labels = torch.ones_like(real_validity).to(device)
    d_real_loss = adversarial_loss(real_validity, real_labels)
    print(f"  Real loss: {d_real_loss.item():.6f}")
    print(f"  Real validity: mean={real_validity.mean().item():.4f}")
    
    print("\nForward pass on fake images...")
    fake_images = generator(lr_images).detach()  # Detach to avoid generator gradients
    fake_validity = discriminator(fake_images)
    fake_labels = torch.zeros_like(fake_validity).to(device)
    d_fake_loss = adversarial_loss(fake_validity, fake_labels)
    print(f"  Fake loss: {d_fake_loss.item():.6f}")
    print(f"  Fake validity: mean={fake_validity.mean().item():.4f}")
    
    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2
    print(f"  Total discriminator loss: {d_loss.item():.6f}")
    
    print("\nBackward pass...")
    d_loss.backward()
    
    has_grads, grad_stats = check_gradients(discriminator, "Discriminator")
    
    if has_grads:
        print("\n✓ Discriminator backward pass SUCCESSFUL")
        print(f"  Gradient norm summary: min={min(s['norm'] for s in grad_stats):.6e}, "
              f"max={max(s['norm'] for s in grad_stats):.6e}")
    else:
        print("\n✗ Discriminator backward pass FAILED - No gradients computed")
    
    print("\nOptimizer step...")
    disc_optimizer.step()
    print("✓ Optimizer step completed")
    
    # ========================================
    # Test 3: Multi-step training simulation
    # ========================================
    print("\n" + "="*60)
    print("TEST 3: Multi-Step Training Simulation")
    print("="*60)
    
    num_steps = 5
    print(f"\nRunning {num_steps} training steps...")
    
    for step in range(num_steps):
        # Train discriminator
        disc_optimizer.zero_grad()
        fake_images = generator(lr_images).detach()
        real_validity = discriminator(hr_images)
        fake_validity = discriminator(fake_images)
        d_loss = (adversarial_loss(real_validity, torch.ones_like(real_validity).to(device)) + 
                  adversarial_loss(fake_validity, torch.zeros_like(fake_validity).to(device))) / 2
        d_loss.backward()
        disc_optimizer.step()
        
        # Train generator
        gen_optimizer.zero_grad()
        fake_images = generator(lr_images)
        fake_validity = discriminator(fake_images)
        g_loss = (content_loss(fake_images, hr_images) + 
                  1e-3 * adversarial_loss(fake_validity, torch.ones_like(fake_validity).to(device)))
        g_loss.backward()
        gen_optimizer.step()
        
        print(f"  Step {step+1}/{num_steps}: G_loss={g_loss.item():.6f}, D_loss={d_loss.item():.6f}")
    
    print("\n✓ Multi-step training completed successfully")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\n✓ All backward propagation tests completed")
    print(f"✓ Device: {device}")
    print(f"✓ Generator trainable parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad):,}")
    print(f"✓ Discriminator trainable parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}")

if __name__ == "__main__":
    test_directml_backprop()
    validate_with_cpu()