import torch 
import torch.nn as nn
import torch.nn.functional as F



# Model aux
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x  # Save input for skip connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out = out + residual
        return out
    
class UpsampleBlock(nn.Module):
    """
    Upsamples by factor of 2 using sub-pixel convolution (PixelShuffle).
    """
    def __init__(self, in_channels, upscale_factor=2):
        super(UpsampleBlock, self).__init__()
        
        # Conv increases channels by upscale_factor^2
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)  # Rearranges channels into spatial dimensions
        out = self.prelu(out)
        return out
    
