import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual 
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.linear = nn.Linear(512 * 16 * 16, latent_dim)

    def forward(self, x):         # [bs, 3, 256, 256]
        x = self.conv_block1(x)   # [bs, 64, 128, 128]
        x = self.conv_block2(x)   # [bs, 128, 64, 64]
        x = self.conv_block3(x)   # [bs, 256, 32, 32]
        x = self.conv_block4(x)   # [bs, 512, 16, 16]
        x = x.view(-1, 512 * 16 * 16)
        x = self.linear(x)
        return x

class ResEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 64)
        self.res_block1 = ResBlock(64, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.res_block2 = ResBlock(128, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.res_block3 = ResBlock(256, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.res_block4 = ResBlock(512, 512)
        self.linear = nn.Linear(512 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.res_block1(x)
        x = self.conv_block2(x)
        x = self.res_block2(x)
        x = self.conv_block3(x)
        x = self.res_block3(x)
        x = self.conv_block4(x)
        x = self.res_block4(x)
        x = x.view(-1, 512 * 16 * 16)
        x = self.linear(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 512 * 16 * 16)
        self.upconv_block1 = UpConvBlock(512, 256)
        self.upconv_block2 = UpConvBlock(256, 128)
        self.upconv_block3 = UpConvBlock(128, 64)
        self.upconv_block4 = UpConvBlock(64, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 16, 16)  # [bs, 512, 16, 16]
        x = self.upconv_block1(x)    # [bs, 256, 32, 32]
        x = self.upconv_block2(x)    # [bs, 128, 64, 64]
        x = self.upconv_block3(x)    # [bs, 64, 128, 128]
        x = self.upconv_block4(x)    # [bs, 3, 256, 256]
        return x


class ResDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 512 * 16 * 16)
        self.res_block1 = ResBlock(512, 512)
        self.upconv_block1 = UpConvBlock(512, 256)
        self.res_block2 = ResBlock(256, 256)
        self.upconv_block2 = UpConvBlock(256, 128)
        self.res_block3 = ResBlock(128, 128)
        self.upconv_block3 = UpConvBlock(128, 64)
        self.res_block4 = ResBlock(64, 64)
        self.upconv_block4 = UpConvBlock(64, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 16, 16)  # [bs, 512, 16, 16]
        x = self.res_block1(x)
        x = self.upconv_block1(x)    # [bs, 256, 32, 32]
        x = self.res_block2(x)
        x = self.upconv_block2(x)    # [bs, 128, 64, 64]
        x = self.res_block3(x)
        x = self.upconv_block3(x)    # [bs, 64, 128, 128]
        x = self.res_block4(x)
        x = self.upconv_block4(x)    # [bs, 3, 256, 256]
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSampler(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 512 * 16 * 16)
        self.upsample_block1 = UpSampleBlock(512, 256)
        self.upsample_block2 = UpSampleBlock(256, 128)
        self.upsample_block3 = UpSampleBlock(128, 64)
        self.upsample_block4 = UpSampleBlock(64, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 16, 16)    # [bs, 512, 16, 16]
        x = self.upsample_block1(x)    # [bs, 256, 32, 32]
        x = self.upsample_block2(x)    # [bs, 128, 64, 64]
        x = self.upsample_block3(x)    # [bs, 64, 128, 128]
        x = self.upsample_block4(x)    # [bs, 3, 256, 256]
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


# class ResAutoEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.encoder = ResEncoder(latent_dim)
#         self.decoder = ResDecoder(latent_dim)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return torch.sigmoid(x)


# class UpSampleAutoEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.encoder = ConvEncoder(latent_dim)
#         self.decoder = UpSampler(latent_dim)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return torch.sigmoid(x)


if __name__ == "__main__":

    pass
