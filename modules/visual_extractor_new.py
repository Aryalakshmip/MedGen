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


# Main Visual Extractor
class VisualExtractor(nn.Module):
    def __init__(self):
        super(VisualExtractor, self).__init__()

        # Load ResNet-101 backbone (pretrained)
        backbone = models.resnet101(pretrained=True)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)  # Conv2
        self.layer2 = backbone.layer2  # Conv3
        self.layer3 = backbone.layer3  # Conv4
        self.layer4 = backbone.layer4  # Conv5

        # Deformable and Semantic Networks
        self.deformable_net = DeformableNetwork()  # Takes Conv3 features as input
        self.semantic_net = SemanticNetwork()     # Takes Conv5 features as input

        # Final 1x1 convolution to reduce concatenated channels to 256
        self.final_conv = nn.Conv2d(768, 256, kernel_size=1)  # 512 (Deformable) + 256 (Semantic)

        # Average Pooling for global features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        """
        Args:
            images: Tensor of shape [batch_size, 2, 3, H, W]
                    Each input batch contains two images (views) per sample.
        Returns:
            patch_feats: Final fused features reshaped to [batch_size, N, 256] where N = H'' * W''
            avg_feats: Global average pooled features of shape [batch_size, 256]
        """
        # Combine the two input images by averaging them
        images_combined = torch.mean(images, dim=1)  # Shape: [batch_size, 3, H, W]

        # Step 1: Feature extraction using ResNet-101
        x = self.layer1(images_combined)  # Conv2 output
        conv3_feats = self.layer2(x)      # Conv3 output
        conv4_feats = self.layer3(conv3_feats)  # Conv4 output
        conv5_feats = self.layer4(conv4_feats)  # Conv5 output

        # Step 2: Deformable Network (from Conv3 features)
        deform_feats = self.deformable_net(conv3_feats)  # Shape: [batch_size, 512, H', W']

        # Step 3: Semantic Network (from Conv5 features)
        semantic_feats = self.semantic_net(conv5_feats)  # Shape: [batch_size, 256, H'', W'']

        # Step 4: Interpolate deformable features to match semantic feature size
        deform_feats_interp = F.interpolate(
            deform_feats, size=semantic_feats.shape[2:], mode='bilinear', align_corners=False
        )  # Shape: [batch_size, 512, H'', W'']

        # Step 5: Concatenate deformable and semantic features
        concatenated_feats = torch.cat((deform_feats_interp, semantic_feats), dim=1)  # Shape: [batch_size, 768, H'', W'']

        # Step 6: Reduce channels to 256 using 1x1 convolution
        fused_feats = self.final_conv(concatenated_feats)  # Shape: [batch_size, 256, H'', W'']

        # Step 7: Flatten spatial dimensions and permute
        batch_size, channels, height, width = fused_feats.shape
        patch_feats = fused_feats.view(batch_size, channels, -1).permute(0, 2, 1)  # Shape: [batch_size, H'' * W'', 256]

        # Step 8: Global average pooling for global features
        avg_feats = self.avg_pool(fused_feats).squeeze(-1).squeeze(-1)  # Shape: [batch_size, 256]

        return patch_feats, avg_feats