# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv_2d(nn.Module):
    """
    Convolution 3 x 3
    => 
    [BN] 
    => 
    Leaky ReLU
    Convolution 3 x 3 
    => 
    [BN] 
    => 
    Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#AdaptiveAvgPool2d
            DoubleConv_2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_2d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv_2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv_2d(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class OutConv_2d(nn.Module):
    "Output Layer Convolution"
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)


class Unet_2d(nn.Module):
    "2D U-Net network"
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(Unet_2d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_2d(n_channels, 64)

        self.down1 = Down_2d(64, 128)
        self.down2 = Down_2d(128, 256)
        self.down3 = Down_2d(256, 512)             
        self.down4 = Down_2d(512, 1024 // 2)

        self.up1 = Up_2d(1024, 512 // 2, bilinear)
        self.up2 = Up_2d(512, 256 // 2, bilinear)
        self.up3 = Up_2d(256, 128 // 2, bilinear)
        self.up4 = Up_2d(128, 64, bilinear)
        self.outc = OutConv_2d(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

class DoubleConv_3d(nn.Module):
    """
    Convolution 3 x 3
    => 
    [BN] 
    => 
    Leaky ReLU
    Convolution 3 x 3 
    => 
    [BN] 
    => 
    Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True),
            nn.Conv3d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_3d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),#AdaptiveAvgPool2d
            DoubleConv_3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # s = np.array((2, 2, 2)).astype('float')
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
            self.conv = DoubleConv_3d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv_3d(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffT = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
            diffT // 2, diffT - diffT // 2
            ])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class OutConv_3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)


class Unet_3d(nn.Module):
    "3D U-Net network"
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(Unet_3d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_3d(n_channels, 64)

        self.down1 = Down_3d(64, 128)
        self.down2 = Down_3d(128, 256)
        self.down3 = Down_3d(256, 512)             
        self.down4 = Down_3d(512, 1024 // 2)

        self.up1 = Up_3d(1024, 512 // 2, bilinear)
        self.up2 = Up_3d(512, 256 // 2, bilinear)
        self.up3 = Up_3d(256, 128 // 2, bilinear)
        self.up4 = Up_3d(128, 64, bilinear)
        self.outc = OutConv_3d(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
