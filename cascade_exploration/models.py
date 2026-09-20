import os
import torch
import torch.nn as nn
from torchvision import models


# ===================== CBAM Attention Module =====================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially infers attention maps along two separate dimensions: channel and spatial.
    This helps the network focus on salient features and suppress unnecessary background noise.
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 1. Channel Attention Module
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 2. Spatial Attention Module
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        x = x * self.ca(x)
        # Apply spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_map


# ===================== VGG19 + U-Net + CBAM Generator =====================
class Generator(nn.Module):
    """
    Multi-stage Generator architecture integrating VGG19 encoder, U-Net skip connections,
    and CBAM attention mechanisms. Designed to process either 3-channel (Stage 1) or
    6-channel (Stages 2 & 3) inputs dynamically.
    """

    def __init__(self, is_first_stage=True, pretrained_path=None):
        super(Generator, self).__init__()

        # Flexible Input Adapter: Compresses 6-channel input down to 3 channels.
        # Defined globally to ensure consistent architecture, but dynamically applied in forward().
        self.input_adapter = nn.Conv2d(6, 3, kernel_size=1)

        # Load pre-trained VGG19 backbone
        vgg19 = models.vgg19(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            vgg19.load_state_dict(torch.load(pretrained_path))
            print(f"[*] Generator successfully loaded pre-trained VGG19 weights: {pretrained_path}")

        features = list(vgg19.features)

        # Encoder (VGG19 Feature Extractor)
        # Note: Naming convention 'enc' is strictly used here to allow selective
        # weight initialization in train.py (freezing pre-trained encoder weights).
        self.enc1 = nn.Sequential(*features[:4])  # Output channels: 64
        self.enc2 = nn.Sequential(*features[4:9])  # Output channels: 128
        self.enc3 = nn.Sequential(*features[9:18])  # Output channels: 256
        self.enc4 = nn.Sequential(*features[18:27])  # Output channels: 512
        self.enc5 = nn.Sequential(*features[27:36])  # Output channels: 512

        # Decoder with Skip Connections and CBAM Attention
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.cbam5 = CBAM(1024)  # 512 (upsampled) + 512 (skip) = 1024

        self.up4 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
        self.cbam4 = CBAM(512)  # 256 (upsampled) + 256 (skip) = 512

        self.up3 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(256)  # 128 (upsampled) + 128 (skip) = 256

        self.up2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128)  # 64 (upsampled) + 64 (skip) = 128

        # Final output layer mapping back to 3-channel RGB/Grayscale image
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, x):
        # --- Core Innovation: Dynamic Input Channel Handling ---
        # If the input tensor has 6 channels (e.g., face + previous sketch in Stages 2 & 3),
        # utilize the 1x1 convolution adapter to reduce dimensionality to 3.
        # If the input is already 3 channels (Stage 1), bypass the adapter.
        if x.size(1) == 6:
            x = self.input_adapter(x)

        # Encoder Path
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)

        # Decoder Path with Skip Connections and Attention Integration
        out = self.up5(s5)
        out = torch.cat([out, s4], dim=1)  # Skip connection
        out = self.cbam5(out)

        out = self.up4(out)
        out = torch.cat([out, s3], dim=1)  # Skip connection
        out = self.cbam4(out)

        out = self.up3(out)
        out = torch.cat([out, s2], dim=1)  # Skip connection
        out = self.cbam3(out)

        out = self.up2(out)
        out = torch.cat([out, s1], dim=1)  # Skip connection
        out = self.cbam2(out)

        return self.final(out)


# ===================== VGG-Style PatchGAN Discriminator =====================
class Discriminator(nn.Module):
    """
    Discriminator network evaluating the authenticity of the generated sketches.
    Utilizes a PatchGAN-like architecture to penalize structure at the scale of image patches.
    """

    def __init__(self, input_nc=3, ndf=64):
        """
        Args:
            input_nc: Number of input channels (e.g., 3 for a standalone image,
                      or 6 if concatenating face and sketch for conditional discrimination).
                      Note: Set to 3 in train.py for the current framework.
            ndf: Number of discriminator filters in the first conv layer.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input Layer
            nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Hidden Layer 1 (Downsample)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Hidden Layer 2 (Downsample)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Hidden Layer 3 (Feature refinement without downsampling)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer (Produces a 1-channel prediction map)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)