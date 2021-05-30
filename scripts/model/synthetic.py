
"""
Synthetic Data Experiments
"""

## Where to Plot
OUTPUT_DIR = "./data/results/synthetic/"

####################
### Imports
####################

## Standard Libary
import os
import sys
import string
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import pairwise

## Local
from smtm.model import helpers
from smtm.util.helpers import make_directory
from smtm.model import LDA, HDP, DTM, IDTM

####################
### Functions
####################

def _generate_word(v):
    """

    """
    letters = string.ascii_lowercase
    divisor = v // len(letters)
    remainder = v % len(letters)
    char = letters[remainder] * (divisor + 1)
    return char

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
    t1p = np.sin(tau / 4); t1p[t1p<=1e-1] = 1e-1
    t2p = np.cos((tau / 4 + np.pi/4)); t2p[t2p<=1e-1] = 1e-1
    t3p = np.ones_like(tau); t3p[len(t3p)//2:] = 0
    t4p = np.ones_like(tau); t4p[:len(t4p)//2] = 0
    t5p = np.zeros_like(tau); t5p[len(t5p)//4:3*len(t5p)//4] = 1
    t6p = np.exp(tau / 6); t6p[len(t6p)//2:] = t6p[:len(t6p)//2][::-1]
    tp = [t1p,t2p,t3p,t4p,t5p,t6p]
    ## Weights of Each Prevalence
    wp = np.array([.15, .15,.1,.1,.1,.3])
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
        phi[0,k] = stats.multivariate_normal(np.zeros(V), beta_0).rvs()
    for t in range(1, alpha.shape[1]):
        for k in range(alpha.shape[0]):
            phi[t,k] = stats.multivariate_normal(phi[t-1, k], beta_1).rvs()
    ## Transform Parameter Space
    phi_T = helpers.logistic_transform(phi, axis=2, keepdims=True)
    ## Sample Number of Data Points
    n_t = stats.poisson(n_mu).rvs(alpha.shape[1])
    ## Initialize Cache
    X = np.zeros((sum(n_t), V), dtype=int)
    t = np.zeros(sum(n_t), dtype=int)
    theta = np.zeros((sum(n_t), alpha.shape[0]))
    ## Sample Data Incrementally
    current_ind = 0
    for epoch, alpha_t in tqdm(enumerate(alpha.T), total=alpha.shape[1], desc="Data Generator", position=0, leave=True):
        epoch_m = stats.poisson(m_mu).rvs(n_t[epoch])
        epoch_theta = stats.dirichlet(alpha_t + smoothing).rvs(n_t[epoch])
        for d in range(n_t[epoch]):
            ## Get Number of Words in Document
            d_m = max(1,epoch_m[d])
            ## Cache Epoch
            t[current_ind] = epoch
            ## Sample Document Topic Distribution
            theta[current_ind] = epoch_theta[d]
            ## Sample Topics from Distribution
            z = np.random.choice(alpha.shape[0], size=d_m, p=theta[current_ind])
            x = helpers.sample_categorical(phi_T[epoch,z])
            ## Sample Words
            for x_ in x:
                X[current_ind, x_] += 1
            current_ind += 1
    ## Vocabulary
    vocabulary = [_generate_word(v) for v in range(V)]
    ## Construct Data for Return
    data = {
        "data":{
                "X":X,
                "t":t,
                "theta":theta,
                "vocabulary":vocabulary,
                },
        "params":{
                "alpha":alpha,
                "alpha_gen":alpha_gen,
                "phi":phi,
                "phi_T":phi_T}
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
    fig, ax = plt.subplots(data["params"]["phi_T"].shape[1] // 3, 3, figsize=(10,5.8), sharey=True, sharex=True)
    ax = ax.ravel()
    for component in range(max(data["params"]["phi_T"].shape[1], len(ax))):
        if component >= ax.shape[0]:
            continue
        cax = ax[component]
        if component >= data["params"]["phi_T"].shape[1]:
            _ = cax.axis("off")
            continue
        cphi = data["params"]["phi_T"][:,component,:].T
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

def plot_traces(model,
                model_type,
                random_seed=42,
                min_component_support=1):
    """

    """
    ## Set Random Seed
    _ = np.random.seed(random_seed)
    ## Model Awareness
    is_dynamic = isinstance(model, DTM) or isinstance(model, IDTM)
    n_plot_topics = model.theta.shape[1]
    if not isinstance(model, IDTM):
        n_plot_terms = len(model.model.used_vocabs)
    else:
        n_plot_terms = model.V
    n_dims = model.phi.shape[1] if is_dynamic else model.phi.shape[0]
    ## Get Epochs and Params
    trace_epochs = [None] if not is_dynamic else range(model.phi.shape[0])
    trace_params = model.cache_params
    ## Directory Setup Or Early Exit
    if len(trace_params) > 0:
        _ = make_directory(f"{OUTPUT_DIR}{model_type}/trace/", remove_existing=True)
    else:
        return None
    ## Document Traces
    if "theta" in trace_params:
        doc_ids = np.random.choice(model.theta.shape[0],
                                   10,
                                   replace=False)
        for doc_id in tqdm(doc_ids, desc="Document-Topic Trace Plots", file=sys.stdout):
            figure = model.plot_document_trace(doc_id, n_plot_topics)
            if isinstance(figure, tuple) and figure[0] is not None:
                figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/theta_{doc_id}.png", dpi=100)
                plt.close(figure[0])
    ## Acceptance Trace (iDTM only)
    if "acceptance" in trace_params and isinstance(model, IDTM):
        figure = model.plot_acceptance_trace()
        if isinstance(figure, tuple) and figure[0] is not None:
            figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/acceptance.png", dpi=100)
            plt.close(figure[0])
    ## Eta (iDTM only)
    if "eta" in trace_params and isinstance(model, IDTM):
        figure = model.plot_eta_trace()
        if isinstance(figure, tuple) and figure[0] is not None:
            figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/eta.png", dpi=100)
            plt.close(figure[0])
    ## Alpha, Gamma, and Phi Traces
    for epoch in tqdm(trace_epochs, desc="Alpha, Gamma, and Phi Trace Plots", file=sys.stdout, position=0, leave=True):
        epoch_name = epoch if epoch is not None else "all"
        if not any(i in trace_params for i in ["alpha","gamma","phi","eta"]):
            continue
        if "alpha" in trace_params:
            if not isinstance(model, IDTM):
                figure = model.plot_alpha_trace(epoch=epoch,
                                                top_k_topics=n_plot_topics)
                if isinstance(figure, tuple) and figure[0] is not None:
                    figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/alpha_{epoch_name}.png", dpi=100)
                    plt.close(figure[0])
            else:
                figure = model.plot_alpha_trace(epochs=[epoch])
                if isinstance(figure, tuple) and figure[0] is not None:
                    figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/alpha_{epoch_name}.png", dpi=100)
                    plt.close(figure[0])
        if "gamma" in trace_params and isinstance(model, IDTM):
            figure = model.plot_gamma_trace(epochs=[epoch])
            if isinstance(figure, tuple) and figure[0] is not None:
                figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/gamma_{epoch_name}.png", dpi=100)
                plt.close(figure[0])
        if "phi" in trace_params:
            for kdim in tqdm(range(n_dims), desc="Component", file=sys.stdout, position=1, leave=False):
                if isinstance(model, IDTM) and model.m[epoch,kdim] < min_component_support:
                    continue
                if isinstance(model, IDTM) and (epoch < model.K_life[kdim][0] or epoch > model.K_life[kdim][1]):
                    continue
                figure = model.plot_topic_trace(topic_id=kdim,
                                                epoch=epoch,
                                                top_k_terms=n_plot_terms)
                if isinstance(figure, tuple) and figure[0] is not None:
                    figure[0].savefig(f"{OUTPUT_DIR}{model_type}/trace/phi_{epoch_name}_{kdim}.png", dpi=100)
                    plt.close(figure[0])

def plot_evolution(model,
                   model_type):
    """

    """
    ## Model Awareness
    n_plot_topics = model.theta.shape[1]
    if not isinstance(model, IDTM):
        n_plot_terms = len(model.model.used_vocabs)
    else:
        n_plot_terms = model.V
    ## Directory
    _ = make_directory(f"{OUTPUT_DIR}{model_type}/vocabulary_evolution/", remove_existing=True)
    ## Component Mask
    if isinstance(model, IDTM):
        component_mask = (model.m.sum(axis=0) / model.m.sum() > 0.001).nonzero()[0]
        component_mask = set(component_mask)
    ## Cycle Through Components
    for component in tqdm(range(n_plot_topics), desc="Evolution Plots", file=sys.stdout, total=n_plot_topics):
        if isinstance(model, IDTM) and component not in component_mask:
            continue
        figure = model.plot_topic_evolution(topic_id=component,
                                            top_k_terms=n_plot_terms,
                                            top_k_type=np.nanmean)
        if isinstance(figure, tuple) and figure[0] is not None:
            figure[0].savefig(f"{OUTPUT_DIR}{model_type}/vocabulary_evolution/topic_{component}.png",dpi=100)
            plt.close(figure[0])
    
def main():
    """

    """
    ## Model Input
    if len(sys.argv) == 1:
        raise ValueError("Should pass model choice to script.")
    ## Generate Data
    data = generate_data(beta_0=1e-1,
                         beta_1=1e-1,
                         n_mu=250,
                         m_mu=25,
                         gamma=10)
    ## Number of Timepoints
    n_timepoints = max(data["data"]["t"]) + 1
    ## Check Directories
    for d in [OUTPUT_DIR]:
        _ = make_directory(d)
    ## Plot Topic Prevalence
    fig, ax = plot_topic_proportions(data)
    fig.savefig(f"{OUTPUT_DIR}topic_proportions.png",dpi=300)
    plt.close(fig)
    ## Plot Vocabulary Evolution
    fig, ax = plot_vocabulary_evolution(data)
    fig.savefig(f"{OUTPUT_DIR}vocabulary_evolution.png",dpi=300)
    plt.close(fig)
    ## Parse Command Line
    model_type = sys.argv[1]
    model = None
    model_directory = f"{OUTPUT_DIR}{model_type}/"
    ## Create Model Directory
    _ = make_directory(model_directory, remove_existing=True)
    # Fit Models
    if sys.argv[1] == "lda":
        print("Fitting LDA Model")
        model = LDA(vocabulary=data["data"]["vocabulary"],
                    n_iter=50000,
                    n_sample=1000,
                    cache_params=["theta","phi","alpha"],
                    cache_rate=10,
                    verbose=True,
                    jobs=8,
                    k=6,
                    seed=42,
                    alpha=0.01,
                    eta=0.01)
        model = model.fit(data["data"]["X"],
                          checkpoint_location=model_directory,
                          checkpoint_frequency=10000)
        model_infer, model_ll = model.theta, model.ll
    if sys.argv[1] == "hdp":
        print("Fitting HDP Model")
        model = HDP(vocabulary=data["data"]["vocabulary"],
                    n_iter=50000,
                    n_sample=1000,
                    cache_params=["theta","phi","alpha"],
                    cache_rate=10,
                    verbose=True,
                    jobs=8,
                    initial_k=6,
                    alpha=0.01,
                    eta=0.01,
                    threshold=0.01,
                    seed=42)
        model = model.fit(data["data"]["X"],
                          checkpoint_location=model_directory,
                          checkpoint_frequency=10000)
        model_infer, model_ll = model.theta, model.ll
    if sys.argv[1] == "dtm":
        print("Fitting DTM Model")
        model = DTM(vocabulary=data["data"]["vocabulary"],
                    n_iter=50000,
                    n_sample=1000,
                    cache_params=["theta","phi","alpha"],
                    cache_rate=10,
                    verbose=True,
                    jobs=8,
                    t=n_timepoints,
                    k=6,
                    alpha_var=0.01,
                    eta_var=0.01,
                    phi_var=0.01,
                    seed=42)
        model = model.fit(data["data"]["X"],
                          labels=data["data"]["t"],
                          labels_key="timepoint",
                          checkpoint_location=model_directory,
                          checkpoint_frequency=10000)
        model_infer, model_ll = model.theta, model.ll
    if sys.argv[1] == "idtm":
        print("Fitting iDTM Model")
        model = IDTM(vocabulary=data["data"]["vocabulary"],
                     initial_k=6,
                     alpha_0_a=1,
                     alpha_0_b=1,
                     gamma_0_a=1,
                     gamma_0_b=1,
                     sigma_0=10,
                     rho_0=1e-3,
                     delta=8,
                     lambda_0=1000,
                     q=5,
                     t=n_timepoints,
                     q_dim=3,
                     q_var=1e-1,
                     q_weight=0.5,
                     q_type="hmm",
                     alpha_filter=4,
                     gamma_filter=10,
                     n_filter=10,
                     threshold=None,
                     k_filter_frequency=None,
                     batch_size=None,
                     n_iter=2000,
                     n_burn=1,
                     cache_rate=1,
                     cache_params=set(["alpha","phi","acceptance","theta"]),
                     jobs=8,
                     seed=42,
                     verbose=True)
        model = model.fit(data["data"]["X"],
                          data["data"]["t"],
                          checkpoint_location=model_directory,
                          checkpoint_frequency=100)
        model_infer = model.theta
    ## Save Models and Trace Plots
    if model is None:
        raise ValueError("Model selected is not available.")
    ## Save Model
    print("Saving {} Model".format(model_type.upper()))
    _ = model.save(f"{model_directory}/model.joblib")
    ## Vocabulary Evolution
    if isinstance(model, DTM) or isinstance(model, IDTM):
        print("Generating Evolution Plots for {} Model".format(model_type.upper()))
        _ = plot_evolution(model, model_type)
    ## Trace Plots
    print("Generating Trace Plots for {} Model".format(model_type.upper()))
    mmin = 0
    if isinstance(model, IDTM):
        mmin = int(model.m.sum() * 0.001)
    _ = plot_traces(model, model_type, random_seed=42, min_component_support=mmin)
    ## Learned Topic Distribution Over Epochs
    df = pd.DataFrame(np.hstack([data["data"]["t"].reshape(-1,1), model_infer]))
    df_agg = df.groupby(0).mean()
    df_agg = df_agg.apply(lambda x: x/sum(x), axis=1)
    if isinstance(model, IDTM):
        component_mask = ((model.m.sum(axis=0) / model.m.sum()) >= 0.001).nonzero()[0] + 1
        df_agg = df_agg[component_mask]
    ## Make Plot
    fig, ax = plt.subplots(figsize=(10,5.6))
    ax.imshow(df_agg.T, aspect="auto", cmap=plt.cm.Reds)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Component", fontweight="bold")
    ax.set_xticks(range(df_agg.shape[1]))
    ax.set_xticklabels([i-1 for i in df_agg.columns])
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(f"{model_directory}/topic_recovery.png", dpi=100)
    plt.close(fig)

###################
### Execute
###################

if __name__ == "__main__":
    _ = main()