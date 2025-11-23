import torchvision.models as models
import torch.nn as nn 
import torch 

class VGGLoss(nn.Module):
    def __init__(self, feature_layer=36):
        super(VGGLoss, self).__init__()
        
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()
        
    def forward(self, sr_images, hr_images):
        # ✅ Add VGG normalization (ImageNet stats)
        # VGG expects inputs in range [0, 1] with ImageNet normalization
        # But your HR images are in [-1, 1], so rescale first
        sr_images_norm = (sr_images + 1.0) / 2.0  # [-1,1] -> [0,1]
        hr_images_norm = (hr_images + 1.0) / 2.0
        
        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(sr_images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(sr_images.device)
        
        sr_images_norm = (sr_images_norm - mean) / std
        hr_images_norm = (hr_images_norm - mean) / std
        
        # Extract features
        sr_features = self.feature_extractor(sr_images_norm)
        hr_features = self.feature_extractor(hr_images_norm)
        
        # Compute MSE with rescaling factor
        loss = self.mse_loss(sr_features, hr_features) / 12.75  # ✅ Paper's rescaling
        return loss