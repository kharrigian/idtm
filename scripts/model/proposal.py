
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from smtm.model.idtm import GaussianProposal

## Parameters
V = 16
sigma_0 = 10
rho_0 = 1
model_option = "naive_gaussian"
q_var = 1
q_weight = 0.5

## Set Seed
np.random.seed(42)

## Generate Components
phi = np.zeros((36, V))
phi[0] = stats.multivariate_normal(np.zeros(V), sigma_0).rvs()
for t in range(1, phi.shape[0]):
    phi[t] = stats.multivariate_normal(phi[t-1], rho_0).rvs()

## Model Option
if model_option == "hmm":
    ## HMM Model
    g = GaussianHMM(3,
                    covariance_type="tied",
                    min_covar=0,
                    covars_prior=1,
                    covars_weight=100)
    g = g.fit(phi)
    ## Sample
    phi_star = g.sample(phi.shape[0])[0]
elif model_option == "naive_gaussian":
    ## Initialize
    g = GaussianProposal(q_var, q_weight)
    ## Sample
    phi_star = g.generate(phi)
else:
    ## Proposal Not Recognized
    raise ValueError("Model not recognized")

## Show Distribution
fig, ax = plt.subplots(2,2,figsize=(10,5.8))
ax[0,0].imshow(phi.T, aspect="auto")
ax[1,0].imshow(phi_star.T, aspect="auto")
for v in range(phi.shape[1]):
    ax[0,1].plot(phi[:,v], alpha=0.5)
    ax[1,1].plot(phi_star[:,v], alpha=0.5)
fig.tight_layout()
plt.savefig("test.png")
plt.close()

