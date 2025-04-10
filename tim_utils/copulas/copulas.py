import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from itertools import combinations

class Copula:
    """
    A unified implementation of multivariate copulas for synthetic data generation.
    This implementation supports Gaussian copulas, t-copulas, and vine copulas.
    """
    
    def __init__(self, copula_type='gaussian', df=5, verbose=False):
        """
        Initialize the multivariate copula.
        
        Parameters:
        -----------
        copula_type : str, optional
            The type of copula to use. Options: 'gaussian', 't', 'vine'.
        df : int, optional
            Degrees of freedom for t-copula. Only used if copula_type is 't'.   
        verbose : bool, optional
            Whether to print verbose output.
        """
        self.copula_type = copula_type
        self.df = df
        self.correlation_matrix = None
        self.marginals = {}
        self.scaler = StandardScaler()
        self.fitted = False
        self.constant_columns = {}
        self.non_constant_columns = []
        self.verbose = verbose
        
        # For vine copulas
        if copula_type == 'vine':
            self.vine_structure = None
            self.pair_copulas = {}
    
    def fit(self, data):
        """
        Fit the copula to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            The data to fit the copula to.
        """
        if isinstance(data, pd.DataFrame):
            self.columns = list(data.columns)  # Convert to list to avoid Index issues
            data_array = data.values
        else:
            self.columns = [f'var_{i}' for i in range(data.shape[1])]
            data_array = data
            
        # Check for NaN values
        if np.isnan(data_array).any():
            print("Warning: NaN values detected in the data. They will be replaced with column means.")
            # Replace NaN values with column means
            for i in range(data_array.shape[1]):
                col_mean = np.nanmean(data_array[:, i])
                data_array[:, i] = np.nan_to_num(data_array[:, i], nan=col_mean)
        
        # Identify constant and non-constant columns
        self.constant_columns = {}
        self.non_constant_columns = []
        
        for i, col in enumerate(self.columns):
            if np.all(data_array[:, i] == data_array[0, i]):
                self.constant_columns[col] = data_array[0, i]
                if self.verbose:
                    print(f"Column '{col}' is constant with value {data_array[0, i]}. It will be ignored during copula fitting.")
            else:
                self.non_constant_columns.append(col)
        
        if len(self.non_constant_columns) < 2:
            raise ValueError("At least two non-constant columns are required to fit a copula.")
        
        # Extract non-constant columns for copula fitting
        non_constant_indices = [self.columns.index(col) for col in self.non_constant_columns]
        non_constant_data = data_array[:, non_constant_indices]
        
        # Standardize the non-constant data
        data_scaled = self.scaler.fit_transform(non_constant_data)
        
        # Compute the correlation matrix with error handling
        try:
            self.correlation_matrix = np.corrcoef(data_scaled.T)
        except Exception as e:
            print(f"Error computing correlation matrix: {e}")
            print("Using identity matrix as fallback.")
            self.correlation_matrix = np.eye(len(self.non_constant_columns))
        
        # Check for invalid values in the correlation matrix
        if np.isnan(self.correlation_matrix).any() or np.isinf(self.correlation_matrix).any():
            print("Warning: Invalid values in correlation matrix. Replacing with identity matrix.")
            self.correlation_matrix = np.eye(len(self.non_constant_columns))
        
        # Fit marginal distributions for non-constant columns
        for i, col in enumerate(self.non_constant_columns):
            col_idx = self.columns.index(col)
            # Use empirical CDF for each marginal
            self.marginals[col] = {
                'data': data_array[:, col_idx],
                'sorted_indices': np.argsort(data_array[:, col_idx]),
                'sorted_values': np.sort(data_array[:, col_idx])
            }
        
        # For vine copulas, determine the vine structure
        if self.copula_type == 'vine':
            self._fit_vine_copula(data_scaled)
        
        self.fitted = True
        return self
    
    def _fit_vine_copula(self, data_scaled):
        """
        Fit a vine copula to the data.
        
        Parameters:
        -----------
        data_scaled : numpy.ndarray
            The standardized data.
        """
        n_vars = data_scaled.shape[1]
        
        # Determine the vine structure (using a simple approach)
        # In a real implementation, you would use more sophisticated methods
        # to determine the optimal vine structure
        
        # For simplicity, we'll use a C-vine structure
        self.vine_structure = []
        
        # First tree
        first_tree = []
        for i in range(1, n_vars):
            first_tree.append((0, i))
        self.vine_structure.append(first_tree)
        
        # Remaining trees
        for tree in range(1, n_vars - 1):
            tree_edges = []
            for i in range(1, n_vars - tree):
                tree_edges.append((0, i))
            self.vine_structure.append(tree_edges)
        
        # Fit pair copulas for each edge in the vine
        for tree_idx, tree in enumerate(self.vine_structure):
            for edge in tree:
                i, j = edge
                
                # For simplicity, we'll use Gaussian pair copulas
                # In a real implementation, you would select the best pair copula
                # based on the data
                
                # Extract the data for this pair
                if tree_idx == 0:
                    # First tree: use original data
                    u_i = norm.cdf(data_scaled[:, i])
                    u_j = norm.cdf(data_scaled[:, j])
                else:
                    # Higher trees: use transformed data
                    # This is a simplification; in a real implementation,
                    # you would use the h-function to transform the data
                    u_i = norm.cdf(data_scaled[:, i])
                    u_j = norm.cdf(data_scaled[:, j])
                
                # Compute the correlation for this pair with error handling
                try:
                    correlation = np.corrcoef(u_i, u_j)[0, 1]
                    # Check for invalid values
                    if np.isnan(correlation) or np.isinf(correlation):
                        print(f"Warning: Invalid correlation value for pair ({i}, {j}). Using 0 as fallback.")
                        correlation = 0.0
                except Exception as e:
                    print(f"Error computing correlation for pair ({i}, {j}): {e}")
                    print("Using 0 as fallback.")
                    correlation = 0.0
                
                # Store the pair copula parameters
                self.pair_copulas[(tree_idx, i, j)] = {
                    'type': 'gaussian',
                    'correlation': correlation
                }
    
    def _empirical_cdf(self, x, col):
        """
        Compute the empirical CDF for a given column.
        
        Parameters:
        -----------
        x : float
            The value to compute the CDF for.
        col : str
            The column name.
            
        Returns:
        --------
        float
            The value of the empirical CDF.
        """
        marginal = self.marginals[col]
        sorted_values = marginal['sorted_values']
        
        # Find the index where x would be inserted to maintain order
        idx = np.searchsorted(sorted_values, x)
        
        # Compute the empirical CDF value
        return idx / len(sorted_values)
    
    def _empirical_quantile(self, u, col):
        """
        Compute the empirical quantile for a given column.
        
        Parameters:
        -----------
        u : float
            The probability value.
        col : str
            The column name.
            
        Returns:
        --------
        float
            The quantile value.
        """
        marginal = self.marginals[col]
        sorted_values = marginal['sorted_values']
        
        # Compute the index
        idx = int(np.floor(u * len(sorted_values)))
        idx = min(idx, len(sorted_values) - 1)
        
        return sorted_values[idx]
    
    def _gaussian_copula_sample(self, n_samples, random_state=None):
        """
        Generate samples from a Gaussian copula.
        
        Parameters:
        -----------
        n_samples : int
            The number of samples to generate.
        random_state : int, optional
            The random state to use for the random number generator.
            
        Returns:
        --------
        numpy.ndarray
            The generated samples in the copula space (uniform marginals).
        """
        if random_state is not None:
            np.random.seed(random_state)
        # Generate samples from a multivariate normal distribution
        try:
            samples = np.random.multivariate_normal(
                mean=np.zeros(len(self.non_constant_columns)),
                cov=self.correlation_matrix,
                size=n_samples
            )
        except Exception as e:
            print(f"Error generating samples from multivariate normal: {e}")
            print("Using independent standard normal samples as fallback.")
            samples = np.random.normal(0, 1, (n_samples, len(self.non_constant_columns)))
        
        # Transform to uniform marginals using the normal CDF
        uniform_samples = norm.cdf(samples)
        
        return uniform_samples
    
    def _t_copula_sample(self, n_samples, random_state=None):
        """
        Generate samples from a t-copula.
        
        Parameters:
        -----------
        n_samples : int
            The number of samples to generate.
        random_state : int, optional
            The random state to use for the random number generator.
            
        Returns:
        --------
        numpy.ndarray
            The generated samples in the copula space (uniform marginals).
        """
        if random_state is not None:
            np.random.seed(random_state)
        # Generate samples from a multivariate t distribution
        try:
            samples = np.random.multivariate_t(
                loc=np.zeros(len(self.non_constant_columns)),
                shape=self.correlation_matrix,
                df=self.df,
                size=n_samples
            )
        except Exception as e:
            print(f"Error generating samples from multivariate t: {e}")
            print("Using independent t samples as fallback.")
            samples = np.zeros((n_samples, len(self.non_constant_columns)))
            for i in range(len(self.non_constant_columns)):
                samples[:, i] = np.random.standard_t(df=self.df, size=n_samples)
        
        # Transform to uniform marginals using the t CDF
        uniform_samples = t.cdf(samples, df=self.df)
        
        return uniform_samples
    
    def _vine_copula_sample(self, n_samples, random_state=None):
        """
        Generate samples from a vine copula.
        
        Parameters:
        -----------
        n_samples : int
            The number of samples to generate.
        random_state : int, optional
            The random state to use for the random number generator.
            
        Returns:
        --------
        numpy.ndarray
            The generated samples in the copula space (uniform marginals).
        """
        n_vars = len(self.non_constant_columns)
        samples = np.zeros((n_samples, n_vars))
        
        # Generate samples for the first variable (uniform)
        if random_state is not None:
            np.random.seed(random_state)
        samples[:, 0] = np.random.uniform(0, 1, n_samples)
        
        # Generate samples for the remaining variables
        for i in range(1, n_vars):
            # For simplicity, we'll use a Gaussian pair copula for each edge
            # In a real implementation, you would use the appropriate pair copula
            
            # Find the edge in the vine structure
            tree_idx = 0
            edge = (0, i)
            
            # Get the pair copula parameters
            pair_copula = self.pair_copulas[(tree_idx, 0, i)]
            correlation = pair_copula['correlation']
            
            # Generate samples for this variable
            for j in range(n_samples):
                # Generate a sample from the conditional distribution
                # This is a simplification; in a real implementation,
                # you would use the h-function to generate the sample
                
                # Generate a sample from a bivariate normal distribution
                try:
                    z = np.random.multivariate_normal(
                        mean=[0, 0],
                        cov=[[1, correlation], [correlation, 1]],
                        size=1
                    )[0]
                except Exception as e:
                    print(f"Error generating bivariate normal sample: {e}")
                    print("Using independent standard normal samples as fallback.")
                    z = np.random.normal(0, 1, 2)
                
                # Transform to uniform marginals
                u = norm.cdf(z[1])
                
                # Store the sample
                samples[j, i] = u
        
        return samples
    
    def sample(self, n_samples=1000, random_state=None):
        """
        Generate synthetic samples from the fitted copula.
        
        Parameters:
        -----------
        n_samples : int, optional
            The number of samples to generate.
            
        Returns:
        --------
        pandas.DataFrame
            The generated samples.
        """
        if not self.fitted:
            raise ValueError("The copula must be fitted before sampling.")
        
        # Generate samples from the copula (uniform marginals) for non-constant columns
        if self.copula_type == 'gaussian':
            uniform_samples = self._gaussian_copula_sample(n_samples, random_state)
        elif self.copula_type == 't':
            uniform_samples = self._t_copula_sample(n_samples, random_state)
        elif self.copula_type == 'vine':
            uniform_samples = self._vine_copula_sample(n_samples, random_state)
        else:
            raise ValueError(f"Copula type '{self.copula_type}' is not supported.")
        
        # Create a DataFrame with all columns (including constant ones)
        samples_dict = {}
        
        # Add samples for non-constant columns
        for i, col in enumerate(self.non_constant_columns):
            samples = np.zeros(n_samples)
            for j in range(n_samples):
                samples[j] = self._empirical_quantile(uniform_samples[j, i], col)
            samples_dict[col] = samples
        
        # Add constant values for constant columns
        for col, value in self.constant_columns.items():
            samples_dict[col] = np.full(n_samples, value)
        
        # Create DataFrame with columns in the same order as the original data
        samples_df = pd.DataFrame(samples_dict, columns=self.columns)
        
        return samples_df
    
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of the copula.
        """
        if not self.fitted:
            raise ValueError("The copula must be fitted before plotting.")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(self.correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(f'{self.copula_type.capitalize()} Copula Correlation Matrix (Non-Constant Columns Only)')
        plt.xticks(range(len(self.non_constant_columns)), self.non_constant_columns, rotation=45)
        plt.yticks(range(len(self.non_constant_columns)), self.non_constant_columns)
        plt.tight_layout()
        plt.show()
    
    def plot_marginals(self, n_samples=1000):
        """
        Plot the marginal distributions of the copula.
        
        Parameters:
        -----------
        n_samples : int, optional
            The number of samples to generate for the plot.
        """
        if not self.fitted:
            raise ValueError("The copula must be fitted before plotting.")
        
        samples = self.sample(n_samples)
        
        # Only plot non-constant columns
        n_cols = len(self.non_constant_columns)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(15, 15))
        
        for i, col1 in enumerate(self.non_constant_columns):
            for j, col2 in enumerate(self.non_constant_columns):
                if i == j:
                    # Plot histogram for marginal distribution
                    axes[i, j].hist(samples[col1], bins=30, alpha=0.7)
                    axes[i, j].set_title(f'{col1} Marginal')
                else:
                    # Plot scatter plot for bivariate distribution
                    axes[i, j].scatter(samples[col1], samples[col2], alpha=0.5, s=10)
                    axes[i, j].set_xlabel(col1)
                    axes[i, j].set_ylabel(col2)
        
        plt.suptitle(f'{self.copula_type.capitalize()} Copula Marginals (Non-Constant Columns Only)', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_original_vs_synthetic(self, n_samples=1000):
        """
        Plot the original data versus synthetic data for comparison.
        
        Parameters:
        -----------
        n_samples : int, optional
            The number of synthetic samples to generate for the plot.
        """
        if not self.fitted:
            raise ValueError("The copula must be fitted before plotting.")
        
        # Generate synthetic samples
        synthetic_data = self.sample(n_samples)
        
        # Get the original data
        original_data = pd.DataFrame()
        for col in self.columns:
            if col in self.marginals:
                original_data[col] = self.marginals[col]['data']
            elif col in self.constant_columns:
                original_data[col] = np.full(len(self.marginals[self.non_constant_columns[0]]['data']), 
                                           self.constant_columns[col])
        
        # Plot the original data versus synthetic data
        n_cols = len(self.non_constant_columns)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        
        for i, col in enumerate(self.non_constant_columns):
            # Plot original data
            axes[i, 0].hist(original_data[col], bins=30, alpha=0.7, color='blue', label='Original')
            axes[i, 0].set_title(f'{col} - Original Data')
            axes[i, 0].legend()
            
            # Plot synthetic data
            axes[i, 1].hist(synthetic_data[col], bins=30, alpha=0.7, color='red', label='Synthetic')
            axes[i, 1].set_title(f'{col} - Synthetic Data')
            axes[i, 1].legend()
        
        plt.suptitle(f'Original vs Synthetic Data - {self.copula_type.capitalize()} Copula', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Plot scatter plots for pairs of variables
        if n_cols >= 2:
            fig, axes = plt.subplots(n_cols, n_cols, figsize=(15, 15))
            
            for i, col1 in enumerate(self.non_constant_columns):
                for j, col2 in enumerate(self.non_constant_columns):
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
            
            plt.suptitle(f'Original vs Synthetic Data - {self.copula_type.capitalize()} Copula', fontsize=16)
            plt.tight_layout()
            plt.show()
    
    def compare_statistics(self, n_samples=1000):
        """
        Compare statistics between original and synthetic data.
        
        Parameters:
        -----------
        n_samples : int, optional
            The number of synthetic samples to generate for the comparison.
            
        Returns:
        --------
        dict
            A dictionary containing the statistics for original and synthetic data.
        """
        if not self.fitted:
            raise ValueError("The copula must be fitted before comparing statistics.")
        
        # Generate synthetic samples
        synthetic_data = self.sample(n_samples)
        
        # Get the original data
        original_data = pd.DataFrame()
        for col in self.columns:
            if col in self.marginals:
                original_data[col] = self.marginals[col]['data']
            elif col in self.constant_columns:
                original_data[col] = np.full(len(self.marginals[self.non_constant_columns[0]]['data']), 
                                           self.constant_columns[col])
        
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