## 출력 이미지 해상도 (1024, 1024) 를 위한 고해상도 decoder 코드

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=64, image_size=(1024, 1024)):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512 * 8 * 8) # Upsampling to a high dimension
        
        # Upsampling and decoding convolutional layers 
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # Decoding
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  #Decoding
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=1) #Outputting image channels

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = x.view(-1, 512, 8, 8)  # Reshape to match ConvTranspose input
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Outputting between 0 and 1 for pixel values
        return x 
```

## 고해상도 encoder / decoder

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(128 * 8 * 8, latent_dim)  # Assuming output size of conv layers is (8x8) after all pooling

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.relu1(x)
        x = self.batchnorm2(self.conv2(x))
        x = self.relu2(x)
        x = x.view(-1, 128 * 8 * 8)  # Flatten for fully connected layer
        x = self.fc1(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, image_size=(1024, 1024)):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128 * 8 * 8)  # Output size should match input of first convTranspose
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Deconv Layer
        self.batchnorm1 = nn.BatchNorm2d(64) # Applied BatchNorm 
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # Output channels are 3 for RGB
        self.sigmoid = nn.Sigmoid()  # Apply sigmoid to output pixel values (0-1 range)

    def forward(self, z):
        x = self.fc1(z)
        x = x.view(-1, 128, 8, 8) 
        x = self.batchnorm1(self.conv1(x))  # Apply BatchNorm after conv1
        x = self.relu1(x)
        x = self.sigmoid(self.conv2(x))

        return x
```

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
