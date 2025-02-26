import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np

class Generator(nn.Module):
    """
    Generator class for the GAN
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    
    def forward(self, x):
        output = self.model(x)
        # Note: Don't reshape here, it's better to reshape at the point of use
        return output

class Discriminator(nn.Module):
    """
    Discriminator class for the GAN
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # Fix: reshape x here to ensure it has the right shape for the model
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output

def train_gan(batch_size: int = 32, num_epochs: int = 100, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    """
    The method trains a Generative Adversarial Network and is based on:
    https://realpython.com/generative-adversarial-networks/
    
    The Generator network tries to generate convincing images of handwritten digits.
    The Discriminator needs to detect if the image was created by the Generator or if the image is a real image from
    a known dataset (MNIST).
    If both the Generator and the Discriminator are optimized, the Generator is able to create images that are difficult
    to distinguish from real images. This is goal of a GAN.
    
    This code produces the expected results at first attempt at about 50 epochs.
    
    :param batch_size: The number of images to train in one epoch.
    :param num_epochs: The number of epochs to train the gan.
    :param device: The computing device to use. If CUDA is installed and working then 'cuda:0' is chosen
                   otherwise 'cpu' is chosen. Note: Training a GAN on the CPU is very slow.
    """
    # Set up data transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Download and load MNIST dataset
    try:
        train_set = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transform)
    except:
        print("Failed to download MNIST, retrying with different URL")
        # Fallback URLs for MNIST
        torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348ddbd547a1a'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')
        ]
        train_set = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transform)
    
    # Create a data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Sample some real data
    real_samples, mnist_labels = next(iter(train_loader))
    
    # Display real samples
    fig = plt.figure()
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        ax.axis('off')
    
    fig.tight_layout()
    fig.suptitle("Real images")
    plt.savefig('real_images.png')  # Save figure instead of using display
    plt.show()  # Use plt.show() instead of display(fig)
    
    time.sleep(1)  # Reduced sleep time
    
    # Set up training
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    
    # Loss function
    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # Train GAN
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            
            # FIX: Use the actual batch size from the current batch
            current_batch_size = real_samples.shape[0]
            real_samples_labels = torch.ones((current_batch_size, 1)).to(device=device)
            
            # FIX: Use current_batch_size for latent space samples
            latent_space_samples = torch.randn(current_batch_size, 100).to(device=device)
            
            generated_samples = generator(latent_space_samples)
            # FIX: Reshape generated samples to match the shape expected by the discriminator
            generated_samples_for_discriminator = generated_samples.view(current_batch_size, 1, 28, 28)
            generated_samples_labels = torch.zeros((current_batch_size, 1)).to(device=device)
            
            # Concatenate real and generated samples
            all_samples = torch.cat((real_samples, generated_samples_for_discriminator))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
            
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()
            
            # Data for training the generator
            latent_space_samples = torch.randn(current_batch_size, 100).to(device=device)
            
            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            
            # Reshape for the discriminator
            generated_samples_reshaped = generated_samples.view(current_batch_size, 1, 28, 28)
            
            # Get discriminator output for the generated samples
            output_discriminator_generated = discriminator(generated_samples_reshaped)
            
            # We want the discriminator to classify the generated samples as real (1)
            loss_generator = loss_function(
                output_discriminator_generated, 
                torch.ones((current_batch_size, 1)).to(device=device)
            )
            loss_generator.backward()
            optimizer_generator.step()
            
            # Show loss and samples generated periodically
            if n % (128 // batch_size) == 0:  # Adjust frequency based on batch size
                print(f"Epoch: {epoch} Loss D: {loss_discriminator:.2f} Loss G: {loss_generator:.2f}")
                
                # Generate and display samples
                with torch.no_grad():  # Prevent tracking history for visualization
                    test_latent_space_samples = torch.randn(16, 100).to(device=device)
                    generated_samples = generator(test_latent_space_samples).cpu().detach()
                
                fig = plt.figure(figsize=(8, 8))
                for i in range(16):
                    ax = fig.add_subplot(4, 4, i + 1)
                    # Using reshape instead of view to be safe when working with NumPy
                    ax.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                    ax.axis('off')
                
                plt.suptitle(f"Epoch {epoch} - Generated images")
                plt.tight_layout()
                plt.savefig(f'generated_epoch_{epoch}.png')
                
                # Use plt.close() to prevent excessive figure accumulation
                plt.close(fig)
    
    # Generate final sample images
    with torch.no_grad():
        final_latent_samples = torch.randn(16, 100).to(device=device)
        final_generated_samples = generator(final_latent_samples).cpu().detach()
    
    # Display final generated samples
    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(final_generated_samples[i].reshape(28, 28), cmap="gray_r")
        ax.axis('off')
    
    plt.suptitle("Final Generated Images")
    plt.tight_layout()
    plt.savefig('final_generated_images.png')
    plt.show()
    
    return generator, discriminator  # Return the trained models

# Test the fixed function
if __name__ == "__main__":
    # Fix: handling the display function error
    print("Starting GAN training...")
    generator, discriminator = train_gan(batch_size=64, num_epochs=50)
    print("Training complete!")