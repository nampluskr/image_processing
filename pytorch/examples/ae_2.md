## Encoder for VAE

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)     # 128 * 4 * 4
        self.conv4 = ConvBlock(128, 256)    # 256 * 2 * 2
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        # print(x.shape)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar  


# class Encoder(nn.Module):
#     def __init__(self, latent_dim=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.flatten = nn.Flatten()
#         self.linear_mu = nn.Linear(128 * 4 * 4, latent_dim)
#         self.linear_logvar = nn.Linear(128 * 4 * 4, latent_dim)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = torch.relu(self.conv3(x))
#         x = self.pool2(x)
#         x = self.flatten(x)
#         # print(x.shape)
#         mu = self.linear_mu(x)
#         logvar = self.linear_logvar(x)
#         return mu, logvar  
```
