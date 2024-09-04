import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True),)

    def forward(self, x):
        return self.block(x)


class ConvBlock2X(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.conv2 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.conv3 = nn.Sequential(
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.conv4 = nn.Sequential(
            ConvBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.linear = nn.Linear(512 * 16 * 16, latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 512 * 16 * 16)
        x = self.linear(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True),)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 512 * 16 * 16)
        self.upconv1 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(256, 128)
        self.upconv3 = UpConvBlock(128, 64)
        self.upconv4 = UpConvBlock(64, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 16, 16)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


if __name__ == "__main__":

    pass
