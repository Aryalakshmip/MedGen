import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Deformable Network
class DeformableNetwork(nn.Module):
    def __init__(self):
        super(DeformableNetwork, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1)  # Input: Conv3 features
        self.deform_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # Deformable 3x3
        self.deform_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # Deformable 3x3
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)  # Final 1x1 Conv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.deform_conv1(x))
        x = F.relu(self.deform_conv2(x))
        x = self.conv2(x)  # Output: [batch_size, 512, H, W]
        return x


# Semantic Network
class SemanticNetwork(nn.Module):
    def __init__(self):
        super(SemanticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1)  # Input: Conv5 features
        self.dilated_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6)
        self.dilated_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=12, dilation=12)
        self.dilated_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=18, dilation=18)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)  # Final 1x1 Conv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.dilated_conv1(x))
        x = F.relu(self.dilated_conv2(x))
        x = F.relu(self.dilated_conv3(x))
        x = self.conv2(x)  # Output: [batch_size, 256, H, W]
        return x


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
