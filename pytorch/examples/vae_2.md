## AE

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU()  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers (to reduce to latent dimension)
        self.fc1 = nn.Linear(64 * 7 * 7, latent_dim * 2)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        x = self.relu1(self.conv1(x)) 
        x = self.pool(x)  # MaxPooling
        x = self.relu2(self.conv2(x))
        x = self.pool(x) # MaxPooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten for FC layers
        x = self.relu(self.fc1(x))  # Add BatchNorm if desired here
        x = self.fc2(x) 
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 64 * 7 * 7)  # Adjust output size based on input to FC layers in encoder
        self.relu = nn.ReLU()
        
        # Transposed Convolutional Layers (Upsampling)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.relu_up1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, 7, 7)  
        x = self.relu_up1(self.conv_transpose1(x))  # Add BatchNorm if desired here
        x = nn.Sigmoid()(self.conv_transpose2(x)) # Output with sigmoid activation for pixel values
        return x


```

## VAE

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers (to reduce to latent dimension)
        self.fc1 = nn.Linear(64 * 7 * 7, latent_dim * 2)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        x = self.relu1(self.conv1(x))  
        x = self.pool(x)  # MaxPooling
        x = self.relu2(self.conv2(x))
        x = self.pool(x) # MaxPooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten for FC layers
        x = self.relu(self.fc1(x))  # Add BatchNorm if desired here
        mu, logvar = torch.split(self.fc2(x), split_size=latent_dim, dim=1) # Split output for mu and logvariance
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 64 * 7 * 7)  # Adjust output size based on input to FC layers in encoder
        self.relu = nn.ReLU()

        # Transposed Convolutional Layers (Upsampling)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.relu_up1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc1(z).view(-1, 64, 7, 7)  
        x = self.relu_up1(self.conv_transpose1(x))  # Add BatchNorm if desired here
        x = nn.Sigmoid()(self.conv_transpose2(x)) # Output with sigmoid activation for pixel values
        return x
```

## Training Loop for VAE

```python
import torch
from torch import nn
from torch.distributions import Normal

# ... (Encoder and Decoder classes from previous response)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
criterion = nn.BCELoss() # Binary Cross Entropy Loss for pixel reconstruction

def train_vae(dataloader, epochs=10):
    for epoch in range(epochs):
        for idx, (x, _) in enumerate(dataloader):
            x = x.to(device)  # Move to device

            # Encoding
            mu, logvar = encoder(x) 

            # Sampling latent representation
            latent_dist = Normal(torch.zeros_like(mu), torch.exp(logvar)) 
            z = latent_dist.rsample()

            # Decoding 
            recon_x = decoder(z)

            # Loss Calculation
            loss = criterion(recon_x, x)

            # Backpropagation and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Example usage (assuming you have a dataloader ready)
train_vae(dataloader, epochs=50) 
```
