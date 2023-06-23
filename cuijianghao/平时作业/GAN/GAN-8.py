import torch
import torch.nn as nn  # Don't forget this import
from torchvision.utils import save_image
from torch.autograd import Variable

# Redefine the Generator class
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

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the dimensionality of the noise vector
z_dim = 100

# Load the Generator
G = Generator(z_dim).to(device)
G.load_state_dict(torch.load('generator.pth'))
G.eval()  # Set the model to evaluation mode

# Generate images
z = Variable(torch.randn(8, z_dim)).to(device)
with torch.no_grad():
    generated_images = G(z)

# Denormalize and save the images
generated_images = 0.5 * generated_images + 0.5
save_image(generated_images, 'generated_images.png')
