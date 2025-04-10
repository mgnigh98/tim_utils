import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tim_utils import *
from gan import GAN

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.lib._function_base_impl")

print("="*50)
print("GAN Synthetic Data Generation Demo")
print("="*50)

# Load the data
try:
    data = pd.read_parquet("domainKnowledgeML/combined.parquet")
    print("Original data shape:", data.shape)
    print("\nOriginal data head:")
    print(data.head())
    
    # Check for NaN values
    if data.isna().any().any():
        print("\nWarning: NaN values detected in the data.")
        print("NaN values per column:")
        print(data.isna().sum())
    
    # Check for constant columns
    constant_cols = data.columns[data.nunique() == 1]
        
    
    if constant_cols:
        print("\nWarning: The following columns are constant:")
        for col in constant_cols:
            print(f"  - {col}: {data[col].iloc[0]}")
            data = data.drop(columns=constant_cols)
    
except FileNotFoundError:
    print("Could not find the parquet file. Generating synthetic data for demonstration.")
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated variables
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.7 * x1 + 0.3 * np.random.normal(0, 1, n_samples)
    x3 = 0.5 * x1 + 0.5 * x2 + 0.2 * np.random.normal(0, 1, n_samples)
    
    data = pd.DataFrame({
        'var1': x1,
        'var2': x2,
        'var3': x3
    })
    
    print("Generated synthetic data shape:", data.shape)
    print("\nGenerated data head:")
    print(data.head())

# Initialize the GAN
print("\n"*50)
print("Initializing GAN")
print("="*50)

# Get the number of features
n_features = data.shape[1]

# Initialize the GAN with appropriate dimensions
gan = GAN(
    input_dim=n_features,
    output_dim=n_features,
    latent_dim=100,
    hidden_dims={
        'generator': [128, 256, 512],
        'discriminator': [512, 256, 128]
    },
    scaler_type='standard'
)

# Train the GAN
print("\n"*50)
print("Training GAN")
print("="*50)

# Train the GAN with a smaller number of epochs for demonstration
history = gan.train(
    data=data,
    batch_size=64,
    epochs=50,  # Reduced for demonstration
    d_steps=1,
    g_steps=1,
    save_interval=10,
    verbose=True
)

# Plot the training history
print("\n"*50)
print("Plotting Training History")
print("="*50)

gan.plot_training_history()

# Generate synthetic samples
print("\n"*50)
print("Generating Synthetic Samples")
print("="*50)

synthetic_data = gan.generate(n_samples=1000)
print("Synthetic data shape:", synthetic_data.shape)
print("\nSynthetic data head:")
print(synthetic_data.head())

# Plot original vs synthetic data
print("\n"*50)
print("Plotting Original vs Synthetic Data")
print("="*50)

gan.plot_original_vs_synthetic(data, n_samples=1000)

# Compare statistics
print("\n"*50)
print("Comparing Statistics")
print("="*50)

stats = gan.compare_statistics(data, n_samples=1000)

print("\nOriginal data statistics:")
print(stats['original']['mean'])
print("\nSynthetic data statistics:")
print(stats['synthetic']['mean'])

print("\nOriginal data correlation matrix:")
print(stats['original']['correlation'])
print("\nSynthetic data correlation matrix:")
print(stats['synthetic']['correlation'])

# Save the synthetic data
print("\n"*50)
print("Saving Synthetic Data")
print("="*50)

synthetic_data.to_parquet("domainKnowledgeML/synthetic_data_gan.parquet")
print("Synthetic data saved to 'domainKnowledgeML/synthetic_data_gan.parquet'")

# Save the model
print("\n"*50)
print("Saving GAN Model")
print("="*50)

gan.save("domainKnowledgeML/gan_model.pt")
print("GAN model saved to 'domainKnowledgeML/gan_model.pt'")

print("\n"*50)
print("Demo completed successfully!")
print("="*50) 