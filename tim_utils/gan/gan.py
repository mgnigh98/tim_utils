from tim_utils import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.lib._function_base_impl")

class Generator(nn.Module):
    """
    Generator network for the GAN.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Initialize the generator.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input noise vector.
        output_dim : int
            Dimension of the output (number of features).
        hidden_dims : list, optional
            List of hidden layer dimensions. If None, a default architecture is used.
        """
        super(Generator, self).__init__()
        
        # Default architecture if hidden_dims is not provided
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Parameters:
        -----------
        z : torch.Tensor
            Input noise vector.
            
        Returns:
        --------
        torch.Tensor
            Generated data.
        """
        return self.model(z)

class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.
    """
    def __init__(self, input_dim, hidden_dims=None, mode='gan'):
        """
        Initialize the discriminator.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input (number of features).
        hidden_dims : list, optional
            List of hidden layer dimensions. If None, a default architecture is used.
        mode : str, optional
            Mode of the GAN. Options: 'gan' for traditional GAN, 'wgan' for Wasserstein GAN.
        """
        super(Discriminator, self).__init__()
        
        # Default architecture if hidden_dims is not provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        if mode == 'gan':
            layers.append(nn.Linear(hidden_dims[-1], 1))
            layers.append(nn.Sigmoid())  # Output in range [0, 1]
        elif mode == 'wgan':
            layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data.
            
        Returns:
        --------
        torch.Tensor
            Probability that the input is real.
        """
        return self.model(x)

class GAN:
    """
    Generative Adversarial Network for synthetic data generation.
    """
    def __init__(self, features, latent_dim=100, hidden_dims=None, 
                 device=None, scaler_type=None, mode='gan'):
        """
        Initialize the GAN.
        
        Parameters:
        -----------
        features : int
            Number of features in the input data.
        latent_dim : int, optional
            Dimension of the latent space (noise vector).
        hidden_dims : dict, optional
            Dictionary with keys 'generator' and 'discriminator' containing lists of hidden layer dimensions.
        device : torch.device, optional
            Device to use for computation. If None, it will be determined automatically.
        scaler_type : str, optional
            Type of scaler to use for data preprocessing. Options: None, 'standard', 'minmax'.
        mode : str, optional
            Mode of the GAN. Options: 'gan' for traditional GAN, 'wgan' for Wasserstein GAN.
        """
        self.features = features
        self.latent_dim = latent_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_type = scaler_type
        self.mode = mode
        
        # Set hidden dimensions
        if hidden_dims is None:
            self.hidden_dims = {
                'generator': [128, 256, 512],
                'discriminator': [512, 256, 128]
            }
        else:
            self.hidden_dims = hidden_dims
        
        # Initialize the networks
        self.generator = Generator(latent_dim, features, self.hidden_dims['generator']).to(self.device)
        self.discriminator = Discriminator(features, self.hidden_dims['discriminator'], mode=self.mode).to(self.device)
        
        # Initialize the optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Initialize the loss function
        if self.mode == 'gan':
            self.criterion = nn.BCELoss()
        elif self.mode == 'wgan':
            self.criterion = self.wasserstein_loss
        
        # Initialize the scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Scaler type '{scaler_type}' is not supported.")
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'g_acc': [],
            'd_acc': []
        }
    
    def wasserstein_loss(self, y_pred, y_true):
        """
        Wasserstein loss function.
        
        Parameters:
        -----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.
            
        Returns:
        --------
        torch.Tensor
            Loss value.
        """
        return torch.mean(y_true * y_pred)
    
    def preprocess_data(self, data):
        """
        Preprocess the data for training.
        
        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            The data to preprocess.
            
        Returns:
        --------
        torch.Tensor
            The preprocessed data as a PyTorch tensor.
        """
        # Convert to numpy array if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            self.columns = data.columns
        else:
            data_array = data
            self.columns = [f'feature_{i}' for i in range(data.shape[1])]
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data_array)
        
        # Convert to PyTorch tensor
        data_tensor = torch.FloatTensor(data_scaled).to(self.device)
        
        return data_tensor
    
    def train(self, data, batch_size=64, epochs=100, d_steps=1, g_steps=1, 
              save_interval=10, verbose=True):
        """
        Train the GAN.
        
        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            The data to train on.
        batch_size : int, optional
            Batch size for training.
        epochs : int, optional
            Number of epochs to train for.
        d_steps : int, optional
            Number of discriminator training steps per generator step.
        g_steps : int, optional
            Number of generator training steps per discriminator step.
        save_interval : int, optional
            Interval at which to save the model.
        verbose : bool, optional
            Whether to print progress.
            
        Returns:
        --------
        dict
            Training history.
        """
        # Preprocess the data
        if self.scaler_type is not None:
            data_tensor = self.preprocess_data(data)
        else:
            data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Create a DataLoader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            g_accs = []
            d_accs = []
            
            for batch in dataloader:
                real_data = batch[0]
                batch_size = real_data.size(0)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train the discriminator
                for _ in range(d_steps):
                    self.d_optimizer.zero_grad()
                    
                    # Generate fake data
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)
                    fake_data = self.generator(z)
                    
                    # Compute discriminator loss on real data
                    d_real = self.discriminator(real_data)
                    d_real_loss = self.criterion(d_real, real_labels)
                    
                    # Compute discriminator loss on fake data
                    d_fake = self.discriminator(fake_data.detach())
                    d_fake_loss = self.criterion(d_fake, fake_labels)
                    
                    # Total discriminator loss
                    d_loss = d_real_loss + d_fake_loss
                    
                    # Backpropagate and update discriminator
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    # Apply weight clipping for WGAN
                    if self.mode == 'wgan':
                        for p in self.discriminator.parameters():
                            p.data.clamp_(-0.01, 0.01)
                    
                    # Compute discriminator accuracy
                    d_real_acc = (d_real > 0.5).float().mean().item()
                    d_fake_acc = (d_fake < 0.5).float().mean().item()
                    d_acc = (d_real_acc + d_fake_acc) / 2
                    
                    d_losses.append(d_loss.item())
                    d_accs.append(d_acc)
                
                # Train the generator
                for _ in range(g_steps):
                    self.g_optimizer.zero_grad()
                    
                    # Generate fake data
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)
                    fake_data = self.generator(z)
                    
                    # Compute generator loss
                    g_fake = self.discriminator(fake_data)
                    g_loss = self.criterion(g_fake, real_labels)
                    
                    # Backpropagate and update generator
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    # Compute generator accuracy
                    g_acc = (g_fake > 0.5).float().mean().item()
                    
                    g_losses.append(g_loss.item())
                    g_accs.append(g_acc)
            
            # Update training history
            self.history['g_loss'].append(np.mean(g_losses))
            self.history['d_loss'].append(np.mean(d_losses))
            self.history['g_acc'].append(np.mean(g_accs))
            self.history['d_acc'].append(np.mean(d_accs))
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"D Loss: {self.history['d_loss'][-1]:.4f}, "
                      f"G Loss: {self.history['g_loss'][-1]:.4f}, "
                      f"D Acc: {self.history['d_acc'][-1]:.4f}, "
                      f"G Acc: {self.history['g_acc'][-1]:.4f}")
            
            # Save the model
            if (epoch + 1) % save_interval == 0:
                self.save(f"gan_model_epoch_{epoch+1}.pt")
        
        return self.history
    
    def generate(self, n_samples=1000):
        """
        Generate synthetic samples.
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples to generate.
            
        Returns:
        --------
        pandas.DataFrame
            The generated samples.
        """
        # Set the generator to evaluation mode
        self.generator.eval()
        
        # Generate noise
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        
        # Generate synthetic data
        with torch.no_grad():
            synthetic_data = self.generator(z)
        
        # Convert to numpy array
        synthetic_data = synthetic_data.cpu().numpy()
        
        # Inverse transform the data
        if self.scaler_type is not None:
            synthetic_data = self.scaler.inverse_transform(synthetic_data)
        
        # Convert to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.columns)
        
        return synthetic_df
    
    def save(self, filename):
        """
        Save the model.
        
        Parameters:
        -----------
        filename : str
            Filename to save the model to.
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'scaler': self.scaler,
            'columns': self.columns,
            'history': self.history
        }, filename)
    
    def load(self, filename):
        """
        Load the model.
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        self.columns = checkpoint['columns']
        self.history = checkpoint['history']
    
    def plot_training_history(self):
        """
        Plot the training history.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.history['g_loss'], label='Generator Loss')
        plt.plot(self.history['d_loss'], label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(2, 1, 2)
        plt.plot(self.history['g_acc'], label='Generator Accuracy')
        plt.plot(self.history['d_acc'], label='Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracies')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_original_vs_synthetic(self, original_data, n_samples=1000):
        """
        Plot the original data versus synthetic data for comparison.
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            The original data.
        n_samples : int, optional
            Number of synthetic samples to generate.
        """
        # Generate synthetic samples
        synthetic_data = self.generate(n_samples)
        
        # Get the number of features
        n_features = len(self.columns)
        
        # Plot histograms for each feature
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 5*n_features))
        
        for i, col in enumerate(self.columns):
            # Plot original data
            axes[i, 0].hist(original_data[col], bins=30, alpha=0.7, color='blue', label='Original')
            axes[i, 0].set_title(f'{col} - Original Data')
            axes[i, 0].legend()
            
            # Plot synthetic data
            axes[i, 1].hist(synthetic_data[col], bins=30, alpha=0.7, color='red', label='Synthetic')
            axes[i, 1].set_title(f'{col} - Synthetic Data')
            axes[i, 1].legend()
        
        plt.suptitle('Original vs Synthetic Data - GAN', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Plot scatter plots for pairs of features
        if n_features >= 2:
            fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
            
            for i, col1 in enumerate(self.columns):
                for j, col2 in enumerate(self.columns):
                    if i == j:
                        # Plot histogram for marginal distribution
                        axes[i, j].hist(original_data[col1], bins=30, alpha=0.7, color='blue', label='Original')
                        axes[i, j].hist(synthetic_data[col1], bins=30, alpha=0.7, color='red', label='Synthetic')
                        axes[i, j].set_title(f'{col1} Marginal')
                        axes[i, j].legend()
                    else:
                        # Plot scatter plot for bivariate distribution
                        axes[i, j].scatter(original_data[col1], original_data[col2], alpha=0.5, s=10, color='blue', label='Original')
                        axes[i, j].scatter(synthetic_data[col1], synthetic_data[col2], alpha=0.5, s=10, color='red', label='Synthetic')
                        axes[i, j].set_xlabel(col1)
                        axes[i, j].set_ylabel(col2)
                        axes[i, j].legend()
            
            plt.suptitle('Original vs Synthetic Data - GAN', fontsize=16)
            plt.tight_layout()
            plt.show()
    
    def compare_statistics(self, original_data, n_samples=1000):
        """
        Compare statistics between original and synthetic data.
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            The original data.
        n_samples : int, optional
            Number of synthetic samples to generate.
            
        Returns:
        --------
        dict
            A dictionary containing the statistics for original and synthetic data.
        """
        # Generate synthetic samples
        synthetic_data = self.generate(n_samples)
        
        # Compute statistics
        stats_dict = {
            'original': {
                'mean': original_data.mean(),
                'std': original_data.std(),
                'min': original_data.min(),
                'max': original_data.max(),
                'correlation': original_data.corr()
            },
            'synthetic': {
                'mean': synthetic_data.mean(),
                'std': synthetic_data.std(),
                'min': synthetic_data.min(),
                'max': synthetic_data.max(),
                'correlation': synthetic_data.corr()
            }
        }
        
        return stats_dict

