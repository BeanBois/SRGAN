import torch 
import torch.nn as nn
import torch.nn.functional as F
from .auxillaries import ResidualBlock, UpsampleBlock


class GenerativeNetwork(nn.Module):
    def __init__(self, num_residual_blocks=16, upscale_factor=4, kernel_size = 3, feature_size = 64):
        super(GenerativeNetwork, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(kernel_size, feature_size, kernel_size=kernel_size**2, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(channels=feature_size))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Post-residual convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(feature_size)
        )
        
        # Upsampling layers (for 4x: two 2x upsamples)
        upsample_blocks = []
        for _ in range(upscale_factor // 2):
            upsample_blocks.append(UpsampleBlock(feature_size, upscale_factor=2))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        
        # Final output convolution
        self.conv3 = nn.Conv2d(feature_size, kernel_size, kernel_size=kernel_size**2, padding=4)
        
    def forward(self, x):
        # Initial feature extraction
        out1 = self.conv1(x)
        
        # Residual blocks
        out = self.residual_blocks(out1)
        
        # Post-residual conv
        out2 = self.conv2(out)
        
        # Skip connection from input to post-residual
        out = out1 + out2
        
        # Upsampling
        out = self.upsample_blocks(out)
        
        # Final convolution
        out = self.conv3(out)
        
        return out


class DiscriminatoryNetwork(nn.Module):
    def __init__(self, feature_size = 64, alpha = 0.2):
        super(DiscriminatoryNetwork, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(alpha, inplace=True))
            return layers
        
        # Convolutional layers, "increasing by a factor of 2 from 64 to 512 kernels"
        self.conv_blocks = nn.Sequential(
            # Input: 3 x 96 x 96 (for 96x96 HR patches)
            *discriminator_block(3, feature_size, stride=1, bn=False),
            *discriminator_block(feature_size, feature_size, stride=2),    # -> 64 x 48 x 48
            *discriminator_block(feature_size, feature_size * 2, stride=1),
            *discriminator_block(feature_size * 2, feature_size *2, stride=2),  # -> 128 x 24 x 24
            
            *discriminator_block(feature_size * 2, feature_size * 4, stride=1),
            *discriminator_block(feature_size * 4, feature_size * 4, stride=2),  # -> 256 x 12 x 12
            
            *discriminator_block(feature_size * 4, feature_size * 8, stride=1),
            *discriminator_block(feature_size * 8, feature_size * 8, stride=2),  # -> 512 x 6 x 6
        )
        
        # Dense layers , 
        # "resulting 512 feature maps are followed by two dense layers 
        # and a final sigmoid activation function to obtain a probability 
        # for sample classification"
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling -> 512 x 1 x 1
            nn.Flatten(),
            nn.Linear(feature_size * 8, feature_size * 16),
            nn.LeakyReLU(alpha),
            nn.Linear(feature_size * 16, 1),
            nn.Sigmoid()  # Output probability
        )
        
    def forward(self, x):
        out = self.conv_blocks(x)
        out = self.classifier(out)
        return out