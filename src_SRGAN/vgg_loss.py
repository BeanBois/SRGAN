import torchvision.models as models
import torch.nn as nn 

class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps.
    VGG22: feature_layer=9 (conv2_2, before 2nd maxpool)
    VGG54: feature_layer=36 (conv5_4, before 5th maxpool)
    """
    def __init__(self, feature_layer=36):
        super(VGGLoss, self).__init__()
        
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:feature_layer]
        )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()
        # Rescaling factor from paper (1/12.75 ≈ 0.006)
        self.rescale_factor = 1.0 / 12.75
        
    def forward(self, sr_images, hr_images):
        sr_features = self.feature_extractor(sr_images)
        hr_features = self.feature_extractor(hr_images)
        loss = self.mse_loss(sr_features, hr_features) * self.rescale_factor
        return loss