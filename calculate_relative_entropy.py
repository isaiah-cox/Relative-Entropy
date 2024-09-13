import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# list of sources
sources = ['2MASXJ17253053-4510279','2MFGC9836','CGCG1822.3+2053','IC1141','MCG+2-57-2','NGC5759']
# list of models
models = ['borus','uxclumpy']
# upper and lower bounds of each parameter in the borus02 model
priors_borus = {
    'PhoIndex':[1.4,2.6],
    'logNHtor':[22.,25.5],
    'CFtor':[0.1,1.0],
    'cos_thInc':[0.05,0.95],
    'log_norm':[1e-5,1.],
    'nH_cha':[0.1,500.0],
    'nH':[0.1,500.0],
    'log_c_xmm':[0.1,2.0],
    'nH_xmm':[0.1,500.0],
    'log_c_bat':[0.1,10],
    'log_fs':[1e-5,5e-2]
    }
# upper and lower bounds of each parameter in the UXCLUMPY model
priors_uxclumpy = {
    'PhoIndex':[1.4,2.6],
    'TORsigma':[6.0,84.0],
    'CTKcover':[0.0,0.6],
    'Theta_inc':[0.0,90.0],
    'log_norm':[1e-5,1.],
    'nH_cha':[0.1,500.0],
    'nH':[0.1,500.0],
    'log_c_xmm':[0.1,2.0],
    'nH_xmm':[0.1,500.0],
    'log_c_bat':[0.1,10.0],
    'log_fs':[1e-5,5e-2]
    }

# function to read in the posterior data
def get_posterior(source,model,param):
    posterior_data = pd.read_csv(f'/path/to/sources/directory/{source}/output_{model}/posterior.csv')
    return posterior_data[param]

# function to calculate the relative entropy for a parameter with a loguniform prior distribution
def log_relative_entropy(posterior,a,b):
    kde = gaussian_kde(posterior)
    x = np.linspace(min(posterior), max(posterior), 1000)
    p = kde(x)
    q = 1/(x*np.log10(b/a))
    entropy = p * (np.log2(p/q)) * (x[1] - x[0])
    entropy[entropy<0] = 0
    total_entropy = np.sum(entropy)
    return total_entropy

# function to calculate the relative entropy for a parameter with a uniform prior distribution
def relative_entropy(posterior,a,b):
    kde = gaussian_kde(posterior)
    x = np.linspace(min(posterior), max(posterior), 1000)
    p = kde(x)
    q = 1/(b-a)
    plt.plot(x,p)
    # plt.plot(x,q)
    plt.show()
    entropy = p * (np.log2(p/q)) * (x[1] - x[0])
    entropy[entropy<0] = 0
    total_entropy = np.sum(entropy)
    return total_entropy

# funtion to get the relative entropy of a parameter from the borus02 model
def get_relative_entropy_bor(source,model,param):
    if 'log_' in param:
        posterior = np.power(10, np.array(get_posterior(source, model, param)))
        kld = log_relative_entropy(posterior,priors_borus[param][0],priors_borus[param][1])
    else:
        posterior = np.array(get_posterior(source, model, param))
        kld = relative_entropy(posterior,priors_borus[param][0],priors_borus[param][1])
    return f"{source},{model},'{param}':{kld}"

# funtion to get the relative entropy of a parameter from the UXCLUMPY model
def get_relative_entropy_ux(source,model,param):
    if 'log_' in param:
        posterior = np.power(10, np.array(get_posterior(source, model, param)))
        kld = log_relative_entropy(posterior,priors_uxclumpy[param][0],priors_uxclumpy[param][1])
    else:
        posterior = np.array(get_posterior(source, model, param))
        kld = relative_entropy(posterior,priors_uxclumpy[param][0],priors_uxclumpy[param][1])
    return f"{source},{model},'{param}':{kld}"

# write the relative entropy for each parameter to the file 'relative_entropies.txt'
with open('relative_entropies.txt', 'w') as file:
    # write the borus02 results to the file
    for source in sources:
        # if the source only has one soft X-ray observation
        try:
            file.write(get_relative_entropy_bor(source, models[0], 'nH')+'\n')
        # if the source only has chandra and xmm observations
        except:
            file.write(get_relative_entropy_bor(source, models[0], 'nH_cha')+'\n')
            file.write(get_relative_entropy_bor(source, models[0], 'nH_xmm')+'\n')
        file.write(get_relative_entropy_bor(source, models[0], 'PhoIndex')+'\n')
        # if the source has torus parameters free
        try:
            file.write(get_relative_entropy_bor(source, models[0], 'logNHtor')+'\n')
            file.write(get_relative_entropy_bor(source, models[0], 'CFtor')+'\n')
            file.write(get_relative_entropy_bor(source, models[0], 'cos_thInc')+'\n')
        # if the source has torus parameters fixed
        except:
            pass
        file.write(get_relative_entropy_bor(source, models[0], 'log_norm')+'\n')
        # if the source has scattering fraction free
        try:
            file.write(get_relative_entropy_bor(source, models[0], 'log_fs')+'\n')
        # if the source has scattering fraction fixed
        except:
            pass
        # if the source has xmm observation
        try:
            file.write(get_relative_entropy_bor(source, models[0], 'log_c_xmm')+'\n')
        # if the source does not have xmm observation
        except:
            pass
        file.write(get_relative_entropy_bor(source, models[0], 'log_c_bat')+'\n')
    # write the UXCLUMPY results to the file
    for source in sources:
        try:
            file.write(get_relative_entropy_ux(source, models[1], 'nH')+'\n')
        except:
            file.write(get_relative_entropy_ux(source, models[1], 'nH_cha')+'\n')
            file.write(get_relative_entropy_ux(source, models[1], 'nH_xmm')+'\n')
        file.write(get_relative_entropy_ux(source, models[1], 'PhoIndex')+'\n')
        try:
            file.write(get_relative_entropy_ux(source, models[1], 'TORsigma')+'\n')
            file.write(get_relative_entropy_ux(source, models[1], 'CTKcover')+'\n')
            file.write(get_relative_entropy_ux(source, models[1], 'Theta_inc')+'\n')
        except:
            pass
        file.write(get_relative_entropy_ux(source, models[1], 'log_norm')+'\n')
        try:
            file.write(get_relative_entropy_ux(source, models[1], 'log_fs')+'\n')
        except:
            pass
        try:
            file.write(get_relative_entropy_ux(source, models[1], 'log_c_xmm')+'\n')
        except:
            pass
        file.write(get_relative_entropy_ux(source, models[1], 'log_c_bat')+'\n')
