########################################
# File: hmc.py
# Author: Guinness Chen
# Date: 06/09/2024
# Description: Implementation of the Hamiltonian Monte Carlo algorithm 
########################################

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import math
import scipy.stats as stats
import argparse

# define constants
N = 10000


"""
Plan:

We need to write expressions for the potential energy and the kinetic energy of the system.
The potential energy is the negative log of the posterior distribution.
The kinetic energy is gaussian.
The Hamiltonian is the sum of the potential and kinetic energies.
The Hamiltonian dynamics are simulated using leapfrog integration.
The Metropolis-Hastings acceptance criterion is used to accept or reject the proposed state.
The algorithm is run for a number of iterations and the samples are stored.

TODO:
(1) We need to write expressions for the derivatives of the potential energy.
(2) Our model is as follows:
    - Our parameter is theta, which has 6 components (mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau).
    - The data consists of 24 2D samples.
    - There are 4 groups of samples, each with a different distribution.
    - Group 1 has a bivariate normal distribution with mean (mu_1, mu_2) and covariance matrix [[sigma_squared, 0], [0, sigma_squared]].
    - Group 2 has a bivariate normal distribution with mean (gamma_1, gamma_2) and covariance matrix [[sigma_squared, 0], [0, sigma_squared]].
    - Group 3 is a 50/50 mixture of the distributions of groups 1 and 2. So the mean is (mu_1 + gamma_1)/2 and the covariance matrix is [[sigma_squared, 0], [0, sigma_squared]].
    - Group 4 is a mixture of the distributions of groups 1 and 2, with the weights of the two distributions being tau and 1 - tau respectively. So the mean is tau * mu + (1 - tau) * gamma and the covariance matrix is [[sigma_squared, 0], [0, sigma_squared]].
    - The likelihood of the data given the parameters is the product of the likelihoods of the 24 samples. Note that the class labels are known, so we don't need to marginalize over them.

    - The prior distribution of the parameters is as follows:
        - mu_1, mu_2, gamma_1, gamma_2 ~ improper uniform on the real number line
        - sigma_squared is proportional to 1 / sigma_square, where 0.1 <= sigma_squared <= 10
        - tau ~ Uniform(0, 1)

    - The posterior distribution of the parameters is proportional to the product of the likelihood and the prior.
    - The negative log of the posterior distribution is the potential energy of the system.
    - The kinetic energy is gaussian.
        - i.e. we sample the momentum from a unit normal distribution.
"""

#---------------------------------------------------------
# Expressions for the potential energy
#---------------------------------------------------------

def log_p_y_given_theta(y, t, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau):
    if t == 1:
        term_1 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[0] - mu_1)**2 / sigma_squared         
        term_2 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[1] - mu_2)**2 / sigma_squared
        return term_1 + term_2
    elif t == 2:
        term_1 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[0] - gamma_1)**2 / sigma_squared
        term_2 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[1] - gamma_2)**2 / sigma_squared
        return term_1 + term_2
    elif t == 3:
        term_1 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[0] - (mu_1 + gamma_1) / 2)**2 / sigma_squared
        term_2 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[1] - (mu_2 + gamma_2) / 2)**2 / sigma_squared
        return term_1 + term_2
    elif t == 4:
        term_1 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[0] - tau * mu_1 - (1 - tau) * gamma_1)**2 / sigma_squared
        term_2 = -0.5 * np.log(2 * np.pi * sigma_squared) - 0.5 * (y[1] - tau * mu_2 - (1 - tau) * gamma_2)**2 / sigma_squared
        return term_1 + term_2

def log_p_theta(mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau):
    if sigma_squared < 0.1 or sigma_squared > 10:
        return -np.inf
    else:
        return -np.log(sigma_squared)

def log_posterior(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau):
    log_likelihood = 0
    for y, t in data:
        log_likelihood += log_p_y_given_theta(y, t, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau)
    return log_likelihood + log_p_theta(mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau)

def potential_energy(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau):
    return -log_posterior(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau)

#---------------------------------------------------------
# Expressions for the derivatives of the potential energy
#---------------------------------------------------------

# Group 1 derivatives

def group_1_dlogy_dmu_1(y, mu_1, sigma_squared):
    return (y[0] - mu_1) / sigma_squared

def group_1_dlogy_dmu_2(y, mu_2, sigma_squared):
    return (y[1] - mu_2) / sigma_squared

def group_1_dlogy_dsigma_squared(y, mu_1, mu_2, sigma_squared): 
    term_1 = (y[0] - mu_1)**2 / (2 * sigma_squared**2)
    term_2 = (y[1] - mu_2)**2 / (2 * sigma_squared**2)
    return -1 / sigma_squared + term_1 + term_2

# Group 2 derivatives

def group_2_dlogy_dgamma_1(y, gamma_1, sigma_squared):
    return (y[0] - gamma_1) / sigma_squared

def group_2_dlogy_dgamma_2(y, gamma_2, sigma_squared):
    return (y[1] - gamma_2) / sigma_squared

def group_2_dlogy_dsigma_squared(y, gamma_1, gamma_2, sigma_squared):
    term_1 = (y[0] - gamma_1)**2 / (2 * sigma_squared**2)
    term_2 = (y[1] - gamma_2)**2 / (2 * sigma_squared**2)
    return -1 / sigma_squared + term_1 + term_2

# Group 3 derivatives

def group_3_dlogy_dmu_1(y, mu_1, gamma_1, sigma_squared):
    return ((y[0] - (mu_1 + gamma_1) / 2) / sigma_squared) / 2

def group_3_dlogy_dmu_2(y, mu_2, gamma_2, sigma_squared):
    return ((y[1] - (mu_2 + gamma_2) / 2) / sigma_squared) / 2

def group_3_dlogy_dgamma_1(y, mu_1, gamma_1, sigma_squared):
    return ((y[0] - (mu_1 + gamma_1) / 2) / sigma_squared) / 2

def group_3_dlogy_dgamma_2(y, mu_2, gamma_2, sigma_squared):
    return ((y[1] - (mu_2 + gamma_2) / 2) / sigma_squared) / 2

def group_3_dlogy_dsigma_squared(y, mu_1, mu_2, gamma_1, gamma_2, sigma_squared):
    term_1 = (y[0] - (mu_1 + gamma_1) / 2)**2 / (2 * sigma_squared**2)
    term_2 = (y[1] - (mu_2 + gamma_2) / 2)**2 / (2 * sigma_squared**2)
    return -1 / sigma_squared + term_1 + term_2

# Group 4 derivatives

def group_4_dlogy_dmu_1(y, mu_1, gamma_1, tau, sigma_squared):
    return ((y[0] - tau * mu_1 - (1 - tau) * gamma_1) / sigma_squared) * tau

def group_4_dlogy_dmu_2(y, mu_2, gamma_2, tau, sigma_squared):
    return ((y[1] - tau * mu_2 - (1 - tau) * gamma_2) / sigma_squared) * tau

def group_4_dlogy_dgamma_1(y, mu_1, gamma_1, tau, sigma_squared):
    return ((y[0] - tau * mu_1 - (1 - tau) * gamma_1) / sigma_squared) * (1 - tau)

def group_4_dlogy_dgamma_2(y, mu_2, gamma_2, tau, sigma_squared):
    return ((y[1] - tau * mu_2 - (1 - tau) * gamma_2) / sigma_squared) * (1 - tau)

def group_4_dlogy_dsigma_squared(y, mu_1, mu_2, gamma_1, gamma_2, tau, sigma_squared):
    term_1 = (y[0] - tau * mu_1 - (1 - tau) * gamma_1)**2 / (2 * sigma_squared**2)
    term_2 = (y[1] - tau * mu_2 - (1 - tau) * gamma_2)**2 / (2 * sigma_squared**2)
    return -1 / sigma_squared + term_1 + term_2

# Prior derivatives

def d_prior_dmu_1(mu_1):
    return 0

def d_prior_dmu_2(mu_2):
    return 0

def d_prior_dgamma_1(gamma_1):
    return 0

def d_prior_dgamma_2(gamma_2):
    return 0

def d_prior_dsigma_squared(sigma_squared):
    if 0.1 <= sigma_squared <= 10:
        return -1 / sigma_squared ** 2
    else:
        return 0
    
def d_prior_dtau(tau):
    return 0

# Derivatives of the potential energy

def d_potential_energy(data, mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau):
    dmu_1 = 0
    dmu_2 = 0
    dsigma_squared = 0
    dgamma_1 = 0
    dgamma_2 = 0
    dtau = 0

    for y, t in data:
        if t == 1:
            dmu_1 += -group_1_dlogy_dmu_1(y, mu_1, sigma_squared)
            dmu_2 += -group_1_dlogy_dmu_2(y, mu_2, sigma_squared)
            dsigma_squared += -group_1_dlogy_dsigma_squared(y, mu_1, mu_2, sigma_squared)
        elif t == 2:
            dgamma_1 += -group_2_dlogy_dgamma_1(y, gamma_1, sigma_squared)
            dgamma_2 += -group_2_dlogy_dgamma_2(y, gamma_2, sigma_squared)
            dsigma_squared += -group_2_dlogy_dsigma_squared(y, gamma_1, gamma_2, sigma_squared)
        elif t == 3:
            dmu_1 += -group_3_dlogy_dmu_1(y, mu_1, gamma_1, sigma_squared)
            dmu_2 += -group_3_dlogy_dmu_2(y, mu_2, gamma_2, sigma_squared)
            dgamma_1 += -group_3_dlogy_dgamma_1(y, mu_1, gamma_1, sigma_squared)
            dgamma_2 += -group_3_dlogy_dgamma_2(y, mu_2, gamma_2, sigma_squared)
            dsigma_squared += -group_3_dlogy_dsigma_squared(y, mu_1, mu_2, gamma_1, gamma_2, sigma_squared)
        elif t == 4:
            dmu_1 += -group_4_dlogy_dmu_1(y, mu_1, gamma_1, tau, sigma_squared)
            dmu_2 += -group_4_dlogy_dmu_2(y, mu_2, gamma_2, tau, sigma_squared)
            dgamma_1 += -group_4_dlogy_dgamma_1(y, mu_1, gamma_1, tau, sigma_squared)
            dgamma_2 += -group_4_dlogy_dgamma_2(y, mu_2, gamma_2, tau, sigma_squared)
            dsigma_squared += -group_4_dlogy_dsigma_squared(y, mu_1, mu_2, gamma_1, gamma_2, tau, sigma_squared)

    # Adding derivatives of the prior
    dsigma_squared += -d_prior_dsigma_squared(sigma_squared)

    return np.array([dmu_1, dmu_2, dsigma_squared, dgamma_1, dgamma_2, dtau])

#-------------------------------------
# Hamiltonian Monte Carlo algorithm
#-------------------------------------

# Leapfrog integration
def leapfrog(data, theta, momentum, step_size, n_steps):
    theta_new = np.copy(theta)
    momentum_new = np.copy(momentum)

    # half-step for momentum
    momentum_new -= 0.5 * step_size * d_potential_energy(data, *theta_new)
    
    for _ in range(n_steps):
        # full step for position
        theta_new += step_size * momentum_new
        
        # full step for momentum (except at end of trajectory)
        if _ != n_steps - 1:
            momentum_new -= step_size * d_potential_energy(data, *theta_new)

    # half-step for momentum
    momentum_new -= 0.5 * step_size * d_potential_energy(data, *theta_new)
    
    # Negate momentum to make the proposal symmetric
    momentum_new = -momentum_new
    
    return theta_new, momentum_new

# Hamiltonian Monte Carlo algorithm
def hamiltonian_monte_carlo(data, n_samples, initial_position, path_len, step_size):
    samples = []
    theta = np.copy(initial_position)

    print(theta)

    for _ in range(n_samples):
        momentum_prev = np.random.normal(size=theta.shape)  # Sample the initial momentum
        H_prev = potential_energy(data, *theta) - stats.norm.logpdf(momentum_prev).sum()  # Hamiltonian at initial state

        # Propose a new state using leapfrog integration
        theta_new, momentum_new = leapfrog(data, theta, momentum_prev, step_size, int(path_len / step_size))
        H_new = potential_energy(data, *theta_new) - stats.norm.logpdf(momentum_new).sum()  # Hamiltonian at new state 

        # Metropolis-Hastings acceptance criterion
        print(H_new, H_prev, np.exp(H_prev - H_new))

        if np.random.uniform() < np.exp(H_prev - H_new):
            theta = theta_new  # Accept the new state

        samples.append(np.copy(theta))
    
    return np.array(samples)

#-------------------
# Main function
#-------------------

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


    # set sigma_squared to be the sample variance of group 1 and group 2, averaged
   # sigma_squared = (np.var(group_1_data, axis=0).mean() + np.var(group_2_data, axis=0).mean()) / 2

    sigma_squared = 1
    
    # set tau to be 0.5
    tau = 0.5

    initial_position = np.array([mu_1, mu_2, sigma_squared, gamma_1, gamma_2, tau])

    n_samples = 1000
    path_len = 1.0
    step_size = 0.01

    samples = hamiltonian_monte_carlo(data, n_samples, initial_position, path_len, step_size)
    
    # Save the samples to a CSV file
    np.savetxt('samples.csv', samples, delimiter=',')
    
    # Plotting the samples
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
