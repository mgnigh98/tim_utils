# Multivariate Copulas for Synthetic Data Generation

This repository contains an implementation of multivariate copulas for synthetic data generation. The implementation supports Gaussian copulas, t-copulas, and vine copulas.

## Implementation

The main class is `Copula`, which provides a comprehensive implementation of multivariate copulas for synthetic data generation. This class combines the functionality of both the simple and advanced copula implementations, providing a single, unified interface for all copula types.

## Features

- Empirical marginal distributions
- Support for multiple copula types (Gaussian, t, vine)
- Handling of constant columns and NaN values
- Visualization functions for correlation matrices and marginal distributions
- Comparison of original and synthetic data
- Synthetic data generation with preservation of original column order

## Usage

### Basic Usage

```python
import pandas as pd
from unified_copulas import UnifiedCopula

# Load your data
data = pd.read_parquet("your_data.parquet")

# Initialize and fit the copula
copula = UnifiedCopula(copula_type='gaussian')  # or 't' or 'vine'
copula.fit(data)

# Generate synthetic samples
synthetic_data = copula.sample(n_samples=1000)

# Save the synthetic data
synthetic_data.to_parquet("synthetic_data.parquet")
```

### Advanced Usage

```python
import pandas as pd
from unified_copulas import UnifiedCopula

# Load your data
data = pd.read_parquet("your_data.parquet")

# Initialize and fit a t-copula with specific degrees of freedom
copula = UnifiedCopula(copula_type='t', df=5)
copula.fit(data)

# Generate synthetic samples
synthetic_data = copula.sample(n_samples=1000)

# Visualize the correlation matrix
copula.plot_correlation_matrix()

# Visualize the marginals
copula.plot_marginals(n_samples=1000)

# Compare original and synthetic data
copula.plot_original_vs_synthetic(n_samples=1000)

# Compare statistics
stats = copula.compare_statistics(n_samples=1000)
print(stats['original']['mean'])
print(stats['synthetic']['mean'])
print(stats['original']['correlation'])
print(stats['synthetic']['correlation'])
```

## Demo Scripts

The repository includes a demo script `unified_copula_demo.py` that demonstrates how to use the unified copula implementation with different copula types.

## How Copulas Work

Copulas are functions that join multivariate distribution functions to their one-dimensional marginal distribution functions. They are useful for modeling the dependence structure between random variables.

### Sklar's Theorem

Sklar's theorem states that any multivariate distribution function can be written in terms of its marginals and a copula function:

\[ F(x_1, x_2, ..., x_d) = C(F_1(x_1), F_2(x_2), ..., F_d(x_d)) \]

where \( F \) is the joint distribution function, \( F_i \) are the marginal distribution functions, and \( C \) is the copula function.

### Gaussian Copula

The Gaussian copula is derived from the multivariate normal distribution. It is parameterized by a correlation matrix.

### t-Copula

The t-copula is derived from the multivariate t distribution. It is parameterized by a correlation matrix and degrees of freedom.

### Vine Copula

Vine copulas are a flexible class of multivariate copulas that can model complex dependence structures. They are constructed from a sequence of bivariate copulas.

## References

- Nelsen, R. B. (2006). An Introduction to Copulas. Springer.
- Joe, H. (2014). Dependence Modeling with Copulas. Chapman and Hall/CRC.
- Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula constructions of multiple dependence. Insurance: Mathematics and Economics, 44(2), 182-198. 