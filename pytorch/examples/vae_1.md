## CIFAR-10 Image Generation Autoencoder with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define Hyperparameters
input_size = 32 * 32 * 3  # Image size: 32x32 pixels, 3 channels
latent_dim = 128  # Dimensionality of the latent space
batch_size = 64
learning_rate = 0.001
epochs = 10

# Define Encoder Network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        return x

# Define Decoder Network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32 * 8 * 8)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 32, 8, 8)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

# Initialize Model, Loss Function and Optimizer
encoder = Encoder()
decoder = Decoder()
autoencoder = nn.Sequential(encoder, decoder)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training Loop
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output = autoencoder(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Example Image Generation (for visualization)
with torch.no_grad():
    sample = torch.randn(64, latent_dim)  # Generate random noise in latent space
    generated_images = decoder(sample)
    # ... visualize generated_images ...
```

## CIFAR-10 Image Generation Variational Autoencoder with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define Hyperparameters
input_size = 32 * 32 * 3  # Image size: 32x32 pixels, 3 channels
latent_dim = 128  # Dimensionality of the latent space
batch_size = 64
learning_rate = 0.001
epochs = 10

# Define Encoder Network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 8 * 8, latent_dim * 2)  # Output mean and log variance

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(-1, 32 * 8 * 8)
        return x

# Define Decoder Network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32 * 8 * 8)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 32, 8, 8)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

# Initialize Model, Loss Function and Optimizer
encoder = Encoder()
decoder = Decoder()
vae = nn.Sequential(encoder, decoder)
criterion = nn.MSELoss()  
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Data Loading (Same as Autoencoder Example)

# Training Loop 
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = encoder(data)  # Get mean and log variance from encoder
        mu = outputs[:, :latent_dim]  
        logvar = outputs[:, latent_dim:]

        # Reparameterization Trick
        std = torch.exp(0.5 * logvar) 
        z = mu + std * torch.randn_like(std) 

        reconstructed_images = decoder(z)
        loss = criterion(reconstructed_images, data)  

        loss.backward()  # Backpropagate through the whole network
        optimizer.step()
```

## High Resolution VAE

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50
```

### Decoder

```python
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return self.relu(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super().__init__()
        self.up1 = UpsampleBlock(latent_dim, 256)
        self.up2 = UpsampleBlock(256, 128)
        self.up3 = UpsampleBlock(128, 64)
        self.up4 = UpsampleBlock(64, img_channels)

    def forward(self, x):
       x = self.up1(x)
       x = self.up2(x)
       x = self.up3(x)
       x = self.up4(x) 
       return x
```

### Encoder

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = resnet50(pretrained=True)  # Load pretrained ResNet18
        # Remove the fully connected layer (fc) from ResNet
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x
```

### VAE

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, pretrained=True):
        super().__init__()
        self.encoder = Encoder
        self.latent_dim = latent_dim
        self.decoder = Decoder(latent_dim, img_channels)

    def forward(self, x):
        # Encode the image to get feature map
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        mu = nn.Linear(features.shape[1], self.latent_dim)(features)        # Mean 
        logvar = nn.Linear(features.shape[1], self.latent_dim)(features)    # Log Variance
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)                                     # Sample from standard normal
        z = mu + epsilon * std 
        recon_x = self.decoder(z.unsqueeze(-1).unsqueeze(-1))               # Reshape z to input shape of decoder

        return recon_x, mu, logvar
```


### Training Loop

```python
# Hyperparameters
latent_dim = 128
learning_rate = 1e-3
batch_size = 64
epochs = 50

vae = VAE(latent_dim).cuda()  
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for data in train_loader:
        img, _ = data
        img = img.cuda()

        # Forward pass
        recon_img, mu, logvar = vae(img)
        reconstruction_loss = nn.MSELoss()(recon_img, img)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_divergence

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
```
