########################################
# File: importance_sampling.py
# Author: Guinness Chen
# Date: 06/12/2024
# Description: Implementation of the Importance Sampling algorithm
########################################

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# Helper functions to calculate sums of squared residuals
def calculate_rss(data, mu_1, mu_2, gamma_1, gamma_2, tau):
    rss = 0
    for y, t in data:
        if t == 1:
            rss += np.sum((y - np.array([mu_1, mu_2]))**2)
        elif t == 2:
            rss += np.sum((y - np.array([gamma_1, gamma_2]))**2)
        elif t == 3:
            rss += np.sum((y - 0.5 * (np.array([mu_1, mu_2]) + np.array([gamma_1, gamma_2])))**2)
        elif t == 4:
            rss += np.sum((y - (tau * np.array([mu_1, mu_2]) + (1 - tau) * np.array([gamma_1, gamma_2])))**2)
    return rss

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
def importance_sampling(data, n_samples, initial_position):
    mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau = initial_position
    
    samples = []
    weights = []

    for _ in range(n_samples):
        # Stage 1: Sample sigma_squared and tau
        sigma_squared = np.random.uniform(0.1, 10)
        tau = np.random.uniform(0, 1)

        # Stage 2: Sample mu and gamma
        mu_1 = np.random.normal(0, 1)
        mu_2 = np.random.normal(0, 1)
        gamma_1 = np.random.normal(0, 1)
        gamma_2 = np.random.normal(0, 1)

        # Calculate the log of the target density (unnormalized)
        log_target_density = log_likelihood(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau)
        log_target_density += -np.log(sigma_squared)  # prior for sigma_squared
        log_target_density += 0  # priors for mu, gamma, and tau are uniform

        # Calculate the log of the proposal density
        log_proposal_density = 0  # uniform priors for sigma_squared and tau
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

# Main function
def main(input_file):
    # Load the data
    df = pd.read_csv(input_file)
    data = [(np.array([row['gene1'], row['gene2']]), int(row['group'])) for _, row in df.iterrows()]

    # Set the initial position
    # set mu to be the sample mean of group 1
    group_1_data = np.array([y for y, t in data if t == 1])
   
    mu_1 = np.mean([y[0] for y in group_1_data])
    mu_2 = np.mean([y[1] for y in group_1_data])

    # set gamma to be the sample mean of group 2
    group_2_data = [y for y, t in data if t == 2]
    gamma_1 = np.mean([y[0] for y in group_2_data])
    gamma_2 = np.mean([y[1] for y in group_2_data])

    sigma_squared = 1
    
    # set tau to be 0.5
    tau = 0.5

    initial_position = np.array([mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau])

    n_samples = 1000
    samples, normalized_weights = importance_sampling(data, n_samples, initial_position)
    
    # Save the samples and weights to a CSV file
    np.savetxt('importance_sampling_samples.csv', samples, delimiter=',')
    np.savetxt('importance_sampling_weights.csv', normalized_weights, delimiter=',')

    # Plot weighted histograms for each parameter
    parameter_names = ['mu_1', 'mu_2', 'sigma_squared', 'gamma_1', 'gamma_2', 'tau']
    for i, param in enumerate(parameter_names):
        plt.figure()
        plt.hist(samples[:, i], weights=normalized_weights, bins=30, density=True, alpha=0.6, color='g')
        plt.title(f'Weighted Histogram of {param}')
        plt.xlabel(param)
        plt.ylabel('Density')
        plt.savefig(f'weighted_histogram_{param}.png')

    plt.show()

if __name__ == '__main__':
    # Parse the command line arguments to get the source csv file
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='source csv file')
    args = parser.parse_args()
    # Call the main function
    main(args.input_file)
