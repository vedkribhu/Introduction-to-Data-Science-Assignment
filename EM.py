# Reference used for help https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php

import numpy as np
from scipy.stats import multivariate_normal


c = 1.5
mean1 = [0]*32
mean2 = c*np.random.normal(0.0, 1, 32)
std = np.identity(32)
# data = np.random
data1 = np.random.multivariate_normal(mean1, std, (5000,))
data2 = np.random.multivariate_normal(mean2, std, (5000,))
data = np.concatenate((data2, data1))


prior = [0.5, 0.5]
#Random centers
n = len(data)
mu = np.random.randint(min(data[:,0]), max(data[:,0]), size = (2, 32))
cov = np.zeros((2, 32, 32))
# To avoid calculation of inverse error due to not having full rank issues.
reg_cov = 1e-6*np.identity(32)
for i in range(len(cov)):
    np.fill_diagonal(cov[i], 5)
log_liklihood = []
n_iter = 50

for i in range(n_iter):
    r_ic = np.zeros((10000,len(cov)))
    sum_=0
    for pi_c,mu_c,cov_c in zip(prior,mu,cov):
        cov_c += reg_cov
        sum_ += pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(data) 
    for mean, covariance, p, r in zip(mu, cov, prior, range(len(r_ic[0]))):
        covariance+=reg_cov
        mn = multivariate_normal(mean=mean,cov=covariance)
        r_ic[:, r] = p*mn.pdf(data)/sum_
    mu = []
    cov = []
    prior = []
    log_liklihood = []
    for c in range(len(r_ic[0])):
        m_c = np.sum(r_ic[:, c], axis = 0)
        # print(m_c)
        mu_c = (1/m_c)*np.sum(data*r_ic[:,c].reshape(len(data), 1), axis = 0)
        mu.append(mu_c)
        # print(data.shape)
        cov.append(np.cov(data.T, 
                aweights=(r_ic[:,c]/m_c), 
                bias=True))
        prior.append(m_c/np.sum(r_ic))

print(np.array(mu).shape)
gmm_cent = np.array(mu)
a = np.linalg.norm(gmm_cent[0]-mean1)+np.linalg.norm(gmm_cent[1]-mean2)
b = np.linalg.norm(gmm_cent[1]-mean1)+np.linalg.norm(gmm_cent[0]-mean2)
print(min(a,b))
# posterior = prior
# def iteration():




