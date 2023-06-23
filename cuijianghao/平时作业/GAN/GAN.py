import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Models
class Discriminator(nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        h = self.nonlin1(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = torch.tanh(self.fc2(h))
        out = out.view(out.size(0), 1, 28, 28)
        return out


# Hyperparameters
lr = 0.0002
batch_size = 64
num_epochs = 20
z_dim = 100

# Load FashionMNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.FashionMNIST(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Decide which device we want to run on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize models and optimizers
D = Discriminator().to(device)
G = Generator(z_dim).to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=lr)
optimizerG = torch.optim.Adam(G.parameters(), lr=lr)
criterion = nn.BCELoss()

# Prepare labels for the loss computation outside the loop
real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)

# List to hold the losses
D_losses = []
G_losses = []

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        current_batch_size = images.shape[0]  # Get the current batch size
        images = images.view(current_batch_size, -1).to(device)  # Modify this line

        # Prepare labels for the loss computation inside the loop
        real_labels = torch.ones(current_batch_size, 1).to(device)  # Modify this line
        fake_labels = torch.zeros(current_batch_size, 1).to(device)  # Modify this line

        # Discriminator step
        optimizerD.zero_grad()
        real_outputs = D(images)
        real_loss = criterion(real_outputs, real_labels)

        z = torch.randn(current_batch_size, z_dim).to(device)  # Modify this line
        fake_images = G(z)
        fake_outputs = D(fake_images)
        fake_loss = criterion(fake_outputs, fake_labels)

        D_loss = real_loss + fake_loss
        D_loss.backward()
        optimizerD.step()

        # Generator step
        optimizerG.zero_grad()
        z = torch.randn(current_batch_size, z_dim).to(device)  # Modify this line
        fake_images = G(z)
        outputs = D(fake_images)
        G_loss = criterion(outputs, real_labels)

        G_loss.backward()
        optimizerG.step()

        # Append the losses for plotting
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

    print(f"Epoch [{epoch + 1}/{num_epochs}], Discriminator Loss: {D_loss.item()}, Generator Loss: {G_loss.item()}")

# Save the loss curves as a figure
plt.figure()
plt.plot(D_losses, label='Discriminator loss')
plt.plot(G_losses, label='Generator loss')
plt.legend()
plt.savefig('GAN_losses.png')
plt.close()

# Save the Generator model
torch.save(G.state_dict(), 'generator.pth')

# Save the Discriminator model
torch.save(D.state_dict(), 'discriminator.pth')
