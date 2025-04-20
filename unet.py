
import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = double_conv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), double_conv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = double_conv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = double_conv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)        # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.down3(x3)     # 512
        x5 = self.down4(x4)     # 1024

        x = self.up1(x5)
        x = self._pad_and_concat(x4, x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self._pad_and_concat(x3, x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self._pad_and_concat(x2, x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self._pad_and_concat(x1, x)
        x = self.conv4(x)

        return self.out(x)

    def _pad_and_concat(self, enc_feat, x):
        """Обеспечиваем выравнивание spatial‑размеров"""
        diffY = enc_feat.size()[2] - x.size()[2]
        diffX = enc_feat.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        return torch.cat([enc_feat, x], dim=1)
