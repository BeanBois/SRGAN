import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1, f1=9, f2=5, f3=5, n1=64, n2=32):
        super(SRCNN, self).__init__()
        
        # Layer 1: Patch extraction and representation
        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1, padding=f1//2)
        
        # Layer 2: Non-linear mapping
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2//2)
        
        # Layer 3: Reconstruction
        self.conv3 = nn.Conv2d(n2, num_channels, kernel_size=f3, padding=f3//2)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Layer 1
        out = self.relu(self.conv1(x))
        
        # Layer 2
        out = self.relu(self.conv2(out))
        
        # Layer 3
        out = self.conv3(out)
        
        return out