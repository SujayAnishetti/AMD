import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DU_NetPlusPlus(nn.Module):
    def __init__(self):
        super(DU_NetPlusPlus, self).__init__()
        self.encoder1 = ConvBlock(3, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.decoder4 = ConvBlock(512, 256)
        self.decoder3 = ConvBlock(256, 128)
        self.decoder2 = ConvBlock(128, 64)
        self.decoder1 = ConvBlock(64 + 3, 32)  # 64 from decoder + 3 from original input

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Decoder
        d4 = self.upsample4(e4)
        d4 = self.decoder4(torch.cat([d4, e3], dim=1))

        d3 = self.upsample3(d4)
        d3 = self.decoder3(torch.cat([d3, e2], dim=1))

        d2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([d2, e1], dim=1))

        d1 = self.decoder1(torch.cat([d2, x], dim=1))
        return torch.sigmoid(self.final(d1))
