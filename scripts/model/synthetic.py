
"""
Synthetic Data Experiments
"""

## Where to Plot
PLOT_DIR = "./plots/model/synthetic/"

####################
### Imports
####################

## Standard Libary
import os
import sys

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import pairwise

## Local
from smtm.model import LDA, HDP, DTM

####################
### Functions
####################

def sample_multinomial(p):
    """
    Multinomial sample that gets around floating point error
    with small numbers

    Args:
        p (1d-array): Probability distribution (sums to 1)

    Returns:
        i (int): Sampled index based on probability distribution
    """
    pcum = p.cumsum()
    u = np.random.uniform(0,1)
    for i, v in enumerate(pcum):
        if u < v:
            return i
    return len(p) - 1

def generate_data(gamma=5,
                  V=16,
                  beta_0=10,
                  beta_1=100,
                  n_mu = 100,
                  m_mu = 10,
                  smoothing=1e-10,
                  random_state=42):
    """

    """
    ## Set Random State
    np.random.seed(random_state)
    ## Timepoints
    tau = np.arange(0, 36)
    ## Baseline Prevalence of Each Topic over Time
    t1p = np.sin(tau / 2); t1p[t1p<=1e-1] = 1e-1
    t2p = np.cos((tau / 2 + np.pi/2)); t2p[t2p<=1e-1] = 1e-1
    t3p = np.ones_like(tau); t3p[len(t3p)//2:] = 0
    t4p = np.ones_like(tau); t4p[:len(t4p)//2] = 0
    t5p = np.zeros_like(tau); t5p[len(t5p)//4:3*len(t5p)//4] = 1
    t6p = np.exp(tau / 6); t6p[len(t6p)//2:] = t6p[:len(t6p)//2][::-1]
    tp = [t1p,t2p,t3p,t4p,t5p,t6p]
    ## Weights of Each Prevalence
    wp = np.array([.15, .1,.2,.2,.2,.15])
    ## Prevalence Generator
    alpha_gen = np.zeros((len(tp), len(tau)))
    for k, (w, t) in enumerate(zip(wp, tp)):
        ## Normalize and Weight
        alpha_gen[k] = (t - min(t)) / (max(t) - min(t)) * w
    alpha_gen = (alpha_gen + smoothing) / (alpha_gen + smoothing).sum(axis=0)
    ## Sample Prevalences From Dirichlet Based on Generator
    alpha = np.zeros_like(alpha_gen)
    for epoch, topic_dist in enumerate(alpha_gen.T):
        alpha[:,epoch] = stats.dirichlet(topic_dist * gamma).rvs()[0]
    ## Sample Topic Word Incrementally
    phi = np.zeros((tau.shape[0], alpha.shape[0], V))
    for k in range(alpha.shape[0]):
        phi[0,k] = stats.dirichlet(np.ones(V) * beta_0).rvs()
    for t in range(1, alpha.shape[1]):
        for k in range(alpha.shape[0]):
            phi[t,k] = stats.dirichlet((phi[t-1,k] + smoothing) * beta_1).rvs()[0]
    ## Sample Number of Data Points
    n_t = stats.poisson(n_mu).rvs(alpha.shape[1])
    ## Initialize Cache
    X = np.zeros((sum(n_t), V), dtype=int)
    t = np.zeros(sum(n_t), dtype=int)
    theta = np.zeros((sum(n_t), alpha.shape[0]))
    ## Sample Data Incrementally
    current_ind = 0
    for epoch, alpha_t in tqdm(enumerate(alpha.T), total=alpha.shape[1], desc="Data Generator", position=0, leave=True):
        for d in range(n_t[epoch]):
            ## Get Number of Words in Document
            d_m = stats.poisson(m_mu).rvs()
            ## Cache Epoch
            t[current_ind] = epoch
            ## Sample Document Topic Distribution
            theta[current_ind] = stats.dirichlet(alpha_t + smoothing).rvs()
            ## Sample Topics from Distribution
            z = np.random.choice(alpha.shape[0], size=d_m, p=theta[current_ind])
            ## Sample Words
            for z_ in z:
                x_ = np.random.choice(V, p=phi[epoch,z_])
                X[current_ind,x_] += 1
            current_ind += 1
    ## Construct Data for Return
    data = {
        "data":{
                "X":X,
                "t":t,
                "theta":theta
                },
        "params":{
                "alpha":alpha,
                "alpha_gen":alpha_gen,
                "phi":phi}
    }
    return data

def plot_topic_proportions(data,
                           normalize_generator=False):
    """

    """
    ## Normalize Generator Row-wise
    alpha_gen = data["params"]["alpha_gen"].copy()
    if normalize_generator:
        alpha_gen = alpha_gen / alpha_gen.max(axis=1,keepdims=True)
    ## Topic Proportions over Time
    fig, ax = plt.subplots(1, 2, figsize=(10,5.8), sharey=True)
    a = ax[0].imshow(alpha_gen,aspect="auto",cmap=plt.cm.Reds, vmin=0, vmax=1)
    b = ax[1].imshow(data["params"]["alpha"],aspect="auto",cmap=plt.cm.Reds, vmin=0, vmax=1)
    for i in range(2):
        ax[i].set_xlabel("Epoch", fontweight="bold", fontsize=16)
        ax[i].tick_params(labelsize=14)
    ax[0].set_ylabel("Component", fontweight="bold", fontsize=16)
    cbar = fig.colorbar(b, ax=ax[1])
    cbar.set_label("Mixing Proportion", fontweight="bold", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax[0].set_title("Generating Function", fontweight="bold", fontstyle="italic", fontsize=18, loc="left")
    ax[1].set_title("Sampled Proportion", fontweight="bold", fontstyle="italic", fontsize=18, loc="left")
    fig.tight_layout()
    return fig, ax

def plot_vocabulary_evolution(data):
    """

    """
    ## Vocabulary Evolution
    fig, ax = plt.subplots(data["params"]["phi"].shape[1] // 3, 3, figsize=(10,5.8), sharey=True, sharex=True)
    ax = ax.ravel()
    for component in range(max(data["params"]["phi"].shape[1], len(ax))):
        if component >= ax.shape[0]:
            continue
        cax = ax[component]
        if component >= data["params"]["phi"].shape[1]:
            _ = cax.axis("off")
            continue
        cphi = data["params"]["phi"][:,component,:].T
        m = cax.imshow(cphi, aspect="auto", cmap=plt.cm.Reds, vmin=0, vmax=1)
        cax.tick_params(labelsize=14)
        cax.set_title(f"Component {component}", fontsize=12, fontweight="bold", loc="center")
    fig.suptitle("Vocabulary Evolution", fontsize=18, fontweight="bold", fontstyle="italic")
    fig.text(0.5,0.02,"Epoch",fontweight="bold",fontsize=16,ha="center",va="center")
    fig.text(0.02,0.5,"Term",fontweight="bold",fontsize=16,rotation=90,ha="center",va="center")
    fig.tight_layout()
    fig.subplots_adjust(left=0.075,bottom=0.10)
    return fig, ax

def _pairwise_distances(X, Y, metric):
    """

    """
    assert X.shape[1] == Y.shape[1]
    nx, mx = X.shape
    ny, my = Y.shape
    D = np.zeros((nx, ny))
    for r, row in enumerate(X):
        for c, col in enumerate(Y):
            if c > r:
                continue
            rc_dist = metric(row, col)
            D[r,c] = rc_dist
            D[c,r] = rc_dist
    return D

def align_components(phi_true,
                     phi_pred,
                     nsim=1000):
    """

    """
    ## Shape Alignment
    t_true, k_true, v_true = phi_true.shape
    if len(phi_pred.shape) == 2:
        phi_pred = np.stack([phi_pred for _ in range(t_true)])
    t_pred, k_pred, v_pred = phi_true.shape
    assert t_true == t_pred
    assert v_true == v_pred
    ## Distance Calculation
    D = []
    for epoch in range(t_true):
        D.append(_pairwise_distances(phi_true[epoch],
                                     phi_pred[epoch],
                                     distance.jensenshannon))
    D_mean = np.stack(D).mean(axis=0)
    ## Alignment Simulation (Greedy)
    S_mean = 1 - D_mean
    S_mean = S_mean / S_mean.sum(axis=1, keepdims=True)
    alignments = np.zeros((nsim, k_true), dtype=int)
    k_comps = list(range(k_true))
    for n in range(nsim):
        np.random.shuffle(k_comps)
        matched = set()
        for k in k_comps:
            while True:
                k_assign = np.random.choice(S_mean.shape[1], p=S_mean[k])
                if k_assign not in matched:
                    matched.add(k_assign)
                    alignments[n,k] = k_assign
                    break
    estimated_alignments = [i.most_common(1)[0][0] for i in list(map(Counter, alignments.T))]
    estimated_alignments = dict(zip(range(k_true), estimated_alignments))
    return estimated_alignments

def main():
    """

    """
    ## Generate Data
    data = generate_data()
    ## Check Directories
    for d in [PLOT_DIR]:
        if not os.path.exists(d):
            _ = os.makedirs(d)
    ## Plot Topic Prevalence
    fig, ax = plot_topic_proportions(data)
    fig.savefig(f"{PLOT_DIR}topic_proportions.png",dpi=300)
    plt.close(fig)
    ## Plot Vocabulary Evolution
    fig, ax = plot_vocabulary_evolution(data)
    fig.savefig(f"{PLOT_DIR}vocabulary_evolution.png",dpi=300)
    plt.close(fig)
    ## Fit Models

    lda = LDA(n_iter=1000, n_sample=1000, cache_rate=1, verbose=True, jobs=8, k=6, seed=42, alpha=0.01, eta=0.01)
    lda = lda.fit(data["data"]["X"])
    lda_infer, lda_ll = lda.infer(data["data"]["X"])

    hdp = HDP(n_iter=1000, n_sample=1000, cache_rate=1, verbose=True, jobs=8, initial_k=6, alpha=0.01, eta=0.01, seed=42)
    hdp = hdp.fit(data["data"]["X"])
    hdp_infer, hdp_ll = hdp.infer(data["data"]["X"])

    n_timepoints = max(data["data"]["t"]) + 1
    dtm = DTM(n_iter=1000, n_sample=1000, cache_rate=1, verbose=True, jobs=8, t=n_timepoints, k=6, alpha_var=0.001, phi_var=0.001, eta_var=0.01, seed=42)
    dtm = dtm.fit(data["data"]["X"], labels=data["data"]["t"], labels_key="timepoint")
    dtm_infer, dtm_ll = dtm.infer(data["data"]["X"], labels=data["data"]["t"], labels_key="timepoint")
    
    # inf = data["data"]["theta"]
    # inf = lda_infer
    # inf = hdp_infer
    # inf = dtm_infer

    df = pd.DataFrame(np.hstack([data["data"]["t"].reshape(-1,1), inf]))
    df_agg = df.groupby(0).mean()
    df_agg = df_agg.apply(lambda x: x/sum(x), axis=1)

    plt.imshow(df_agg.T, aspect="auto", cmap=plt.cm.Reds)
    plt.show()

    # v = np.stack([i[1] for i in dtm._theta])
    # v = np.stack([i[1] for i in dtm._phi])

    

###################
### Execute
###################

if __name__ == "__main__":
    _ = main()