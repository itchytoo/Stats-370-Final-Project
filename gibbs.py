########################################
# File: gibbs.py
# Author: Guinness Chen
# Date: 06/09/2024
# Description: Implementation of the Gibbs Sampling algorithm
########################################

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import truncnorm
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

# Truncated normal sample
def truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()

# Truncated inverse-gamma sample
def truncated_invgamma(alpha, beta, low, high):
    while True:
        sample = stats.invgamma.rvs(alpha, scale=beta)
        if low < sample < high:
            return sample

# Gibbs sampling algorithm
def gibbs_sampling(data, n_samples, burn_in, thinning, initial_position):
    mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau = initial_position
    samples = []

    for i in range(n_samples * thinning + burn_in):
        # Sample sigma_squared
        rss = calculate_rss(data, mu_1, mu_2, gamma_1, gamma_2, tau)
        alpha = len(data) / 2
        beta = rss / 2
        sigma_squared = truncated_invgamma(alpha, beta, 0.1, 10)
        
        # Sample mu_1 and mu_2
        G1 = [y for y, t in data if t == 1]
        G3 = [y for y, t in data if t == 3]
        G4 = [y for y, t in data if t == 4]
        a_mu = len(G1) + 0.25 * len(G3) + tau**2 * len(G4)
        b_mu_1 = np.sum([y[0] for y in G1]) + 0.5 * np.sum([y[0] - 0.5 * gamma_1 for y in G3]) + tau * np.sum([y[0] - (1 - tau) * gamma_1 for y in G4])
        b_mu_2 = np.sum([y[1] for y in G1]) + 0.5 * np.sum([y[1] - 0.5 * gamma_2 for y in G3]) + tau * np.sum([y[1] - (1 - tau) * gamma_2 for y in G4])
        mu_1 = np.random.normal(b_mu_1 / a_mu, np.sqrt(sigma_squared / a_mu))
        mu_2 = np.random.normal(b_mu_2 / a_mu, np.sqrt(sigma_squared / a_mu))
        
        # Sample gamma_1 and gamma_2
        G2 = [y for y, t in data if t == 2]
        d_gamma = len(G2) + 0.25 * len(G3) + (1 - tau)**2 * len(G4)
        e_gamma_1 = np.sum([y[0] for y in G2]) + 0.5 * np.sum([y[0] - 0.5 * mu_1 for y in G3]) + (1 - tau) * np.sum([y[0] - tau * mu_1 for y in G4])
        e_gamma_2 = np.sum([y[1] for y in G2]) + 0.5 * np.sum([y[1] - 0.5 * mu_2 for y in G3]) + (1 - tau) * np.sum([y[1] for y in G4])
        gamma_1 = np.random.normal(e_gamma_1 / d_gamma, np.sqrt(sigma_squared / d_gamma))
        gamma_2 = np.random.normal(e_gamma_2 / d_gamma, np.sqrt(sigma_squared / d_gamma))
        
        # Sample tau
        G4 = [y for y, t in data if t == 4]
        A_tau = np.sum([mu_1**2 + gamma_1**2 - 2 * mu_1 * gamma_1 for y in G4])
        A_tau += np.sum([mu_2**2 + gamma_2**2 - 2 * mu_2 * gamma_2 for y in G4])
        B_tau = np.sum([gamma_1**2 + y[0] * mu_1 - y[0] * gamma_1 - mu_1 * gamma_1 for y in G4])
        B_tau += np.sum([gamma_2**2 + y[1] * mu_2 - y[1] * gamma_2 - mu_2 * gamma_2 for y in G4])
        tau_mean = B_tau / A_tau
        tau_var = sigma_squared / A_tau
        tau = truncated_normal(tau_mean, np.sqrt(tau_var), 0, 1)
        
        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples.append([mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau])
    
    return np.array(samples)

# Main function
def main(input_file):
    # Load the data
    df = pd.read_csv(input_file)
    data = [(np.array([row['gene1'], row['gene2']]), int(row['group'])) for _, row in df.iterrows()]

    # Set the initial position
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
    burn_in = 200
    thinning = 4
    samples = gibbs_sampling(data, n_samples, burn_in, thinning, initial_position)
    
    # Save the samples to a CSV file
    np.savetxt('gibbs_samples.csv', samples, delimiter=',')
    
    # Optionally, plot the samples
    samples_df = pd.DataFrame(samples, columns=['mu_1', 'mu_2', 'sigma_squared', 'gamma_1', 'gamma_2', 'tau'])
    sns.pairplot(samples_df)
    plt.show()

if __name__ == '__main__':
    # Parse the command line arguments to get the source csv file
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='source csv file')
    args = parser.parse_args()
    # Call the main function
    main(args.input_file)
