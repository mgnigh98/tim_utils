import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tim_utils import *

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.lib._function_base_impl")

print("="*50)
print("Unified Copula Demo")
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
    constant_cols = []
    for col in data.columns:
        if data[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        print("\nWarning: The following columns are constant:")
        for col in constant_cols:
            print(f"  - {col}: {data[col].iloc[0]}")
    
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

# List of copula types to compare
copula_types = ['gaussian', 't', 'vine']

# Dictionary to store synthetic data for each copula type
synthetic_data_dict = {}

# Fit and sample from each copula type
for copula_type in copula_types:
    print("\n" + "="*50)
    print(f"Fitting {copula_type.capitalize()} Copula")
    print("="*50)
    
    try:
        # Initialize the copula
        if copula_type == 't':
            copula = tim.Copula(copula_type=copula_type, df=5)
        else:
            copula = tim.Copula(copula_type=copula_type)
        
        # Fit the copula
        copula.fit(data)
        print(f"{copula_type.capitalize()} copula fitted successfully.")
        
        # Plot the correlation matrix
        try:
            print("\nPlotting correlation matrix...")
            copula.plot_correlation_matrix()
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
        
        # Generate synthetic samples
        try:
            print("\nGenerating synthetic samples...")
            synthetic_data = copula.sample(n_samples=1000)
            synthetic_data_dict[copula_type] = synthetic_data
            print("Synthetic data shape:", synthetic_data.shape)
            print("\nSynthetic data head:")
            print(synthetic_data.head())
        except Exception as e:
            print(f"Error generating synthetic samples: {e}")
            continue
        
        # Plot the marginals
        try:
            print("\nPlotting marginals...")
            copula.plot_marginals(n_samples=1000)
        except Exception as e:
            print(f"Error plotting marginals: {e}")
        
        # Plot original vs synthetic data
        try:
            print("\nPlotting original vs synthetic data...")
            copula.plot_original_vs_synthetic(n_samples=1000)
        except Exception as e:
            print(f"Error plotting original vs synthetic data: {e}")
        
        # Compare statistics
        try:
            print("\nComparing statistics...")
            stats = copula.compare_statistics(n_samples=1000)
            
            print("\nOriginal data statistics:")
            print(stats['original']['mean'])
            print("\nSynthetic data statistics:")
            print(stats['synthetic']['mean'])
            
            print("\nOriginal data correlation matrix:")
            print(stats['original']['correlation'])
            print("\nSynthetic data correlation matrix:")
            print(stats['synthetic']['correlation'])
        except Exception as e:
            print(f"Error comparing statistics: {e}")
        
        # Save synthetic data
        try:
            synthetic_data.to_parquet(f"domainKnowledgeML/synthetic_data_{copula_type}.parquet")
            print(f"\nSynthetic data saved to 'domainKnowledgeML/synthetic_data_{copula_type}.parquet'")
        except Exception as e:
            print(f"Error saving synthetic data: {e}")
    
    except Exception as e:
        print(f"Error fitting {copula_type} copula: {e}")
        continue

# Compare all copula types
if len(synthetic_data_dict) > 1:
    print("\n" + "="*50)
    print("Comparing Different Copula Types")
    print("="*50)
    
    # Compare correlation matrices
    print("\nCorrelation matrices for different copula types:")
    for copula_type, synthetic_data in synthetic_data_dict.items():
        print(f"\n{copula_type.capitalize()} Copula:")
        print(synthetic_data.corr())
    
    # Compare statistics
    print("\nStatistics for different copula types:")
    for copula_type, synthetic_data in synthetic_data_dict.items():
        print(f"\n{copula_type.capitalize()} Copula:")
        print(synthetic_data.describe())

print("\n" + "="*50)
print("Demo completed successfully!")
print("="*50) 