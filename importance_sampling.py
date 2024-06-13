# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from scipy.stats import gaussian_kde

# Helper function to compute log likelihood
def log_likelihood(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau):
    log_lik = 0
    for y, t in data:
        if t == 1:
            log_lik += -0.5 * (np.sum((y - np.array([mu_1, mu_2]))**2) / sigma_squared + 2 * np.log(2 * np.pi * sigma_squared))
        elif t == 2:
            log_lik += -0.5 * (np.sum((y - np.array([gamma_1, gamma_2]))**2) / sigma_squared + 2 * np.log(2 * np.pi * sigma_squared))
        elif t == 3:
            log_lik += -0.5 * (np.sum((y - 0.5 * (np.array([mu_1, mu_2]) + np.array([gamma_1, gamma_2])))**2) / sigma_squared + 2 * np.log(2 * np.pi * sigma_squared))
        elif t == 4:
            log_lik += -0.5 * (np.sum((y - (tau * np.array([mu_1, mu_2]) + (1 - tau) * np.array([gamma_1, gamma_2])))**2) / sigma_squared + 2 * np.log(2 * np.pi * sigma_squared))
    return log_lik

# Importance Sampling algorithm
def importance_sampling(data, n_samples):
    samples = []
    weights = []

    for _ in range(n_samples):
        # Stage 1: Sample sigma_squared and tau
        sigma_squared = np.random.uniform(0.1, 10)
        tau = np.random.uniform(0, 1)

        # Stage 2: Sample mu and gamma
        mu_1 = np.random.normal(0, 1)
        mu_2 = np.random.normal(7.5, 2)
        gamma_1 = np.random.normal(-0.5, 1)
        gamma_2 = np.random.normal(10.5, 2)

        # Calculate the log of the target density (unnormalized)
        log_target_density = log_likelihood(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau)
        log_target_density += -np.log(sigma_squared)  # prior for sigma_squared

        # Calculate the log of the proposal density
        log_proposal_density = 0
        log_proposal_density += -0.5 * (mu_1**2 + mu_2**2 + gamma_1**2 + gamma_2**2)  # normal proposals for mu and gamma

        # Compute importance weight
        log_weight = log_target_density - log_proposal_density
        weight = np.exp(log_weight)

        samples.append([mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau])
        weights.append(weight)

    # Normalize weights
    weights = np.array(weights)
    normalized_weights = weights / np.sum(weights)

    return np.array(samples), normalized_weights

# Function to plot weighted pair plot
def weighted_pairplot(data, weights, **kwargs):
    g = sns.PairGrid(data, **kwargs)

    # Map upper to scatter plot with weights
    def scatter_func(x, y, **kwargs):
        plt.scatter(x, y, **kwargs)

    g = g.map_upper(scatter_func, alpha=0.5)

    # Map diagonal to weighted histplot
    def hist_func(x, **kwargs):
        plt.hist(x, weights=weights, bins=30, density=True, **kwargs)

    g = g.map_diag(hist_func, color="g", edgecolor="black")

    # Map lower to weighted KDE plot
    def kde_func(x, y, **kwargs):
        # Create weighted KDE
        xy = np.vstack([x, y])
        z = gaussian_kde(xy, weights=weights)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        plt.scatter(x, y, c=z, s=50, cmap='viridis')

    g = g.map_lower(kde_func)

    return g

# Main function
def main(input_file):
    # Load the data
    df = pd.read_csv(input_file)
    data = [(np.array([row['gene1'], row['gene2']]), int(row['group'])) for _, row in df.iterrows()]

    n_samples = 10000
    samples, normalized_weights = importance_sampling(data, n_samples)
    
    # Save the samples and weights to a CSV file
    np.savetxt('importance_sampling_samples.csv', samples, delimiter=',')
    np.savetxt('importance_sampling_weights.csv', normalized_weights, delimiter=',')

    # Convert samples to DataFrame
    samples_df = pd.DataFrame(samples, columns=['mu_1', 'mu_2', 'sigma_squared', 'gamma_1', 'gamma_2', 'tau'])

    # Plot weighted pair plot
    pairplot = weighted_pairplot(samples_df, normalized_weights)
    pairplot.fig.suptitle("Importance Sampling Pair Plot", y=1.02)
    plt.show()

if __name__ == '__main__':
    # Parse the command line arguments to get the source csv file
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='source csv file')
    args = parser.parse_args()
    # Call the main function
    main(args.input_file)
