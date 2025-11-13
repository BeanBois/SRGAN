import torchvision.models as models
import torch.nn as nn 

class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps.
    Computes MSE between feature representations.
    """
    def __init__(self, feature_layer=36):
        super(VGGLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True)
        
        # Extract feature layers up to specified layer
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()
        
    def forward(self, sr_images, hr_images):
        # Extract features
        sr_features = self.feature_extractor(sr_images)
        hr_features = self.feature_extractor(hr_images)
        
        # Compute MSE in feature space
        loss = self.mse_loss(sr_features, hr_features)
        return loss