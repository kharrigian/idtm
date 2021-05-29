
"""
Real Data Modeling
"""

#####################
### Configuration
#####################

## Data
DATA_DIR = "./data/processed/depression/"
DATA_TYPE = "comments"
MIN_DATE = "2019-01-01"
MAX_DATE = "2021-04-01"

## Data Filtering
MIN_VOCAB_DF = 50
MIN_VOCAB_CF = 100
MIN_WORDS_PER_DOCUMENT = 25
MAX_VOCAB_SIZE = 2500

## Topic Model Parameters
MODEL_TYPE = "dtm"
CHECKPOINT_FREQUENCY = 2500
MODEL_PARAMS = {

    # "initial_k":100,
    # "rm_top":0,
    # "alpha":0.1,
    # "eta":0.1,
    # "cache_rate":1,
    # "cache_params":set(["alpha","phi","theta"]),
    # "n_iter":5000,

    # "k":100,
    # "alpha":0.1,
    # "eta":0.1,
    # "cache_rate":1,
    # "cache_params":set(["alpha","phi","theta"]),
    # "n_iter":5000,

    "alpha_var":0.1,
    "eta_var":0.1,
    "phi_var":0.1,
    "rm_top":0,
    "cache_rate":1,
    "cache_params":set(["alpha","phi","theta"]),
    "n_iter":5000,

    # "initial_k":100,
    # "alpha_0_a":1,
    # "alpha_0_b":1,
    # "gamma_0_a":1,
    # "gamma_0_b":1,
    # "sigma_0":1,
    # "rho_0":1e-2,
    # "q":5,
    # "q_dim":2,
    # "q_var":1e-2,
    # "q_weight":0.5,
    # "q_type":"hmm",
    # "alpha_filter":4,
    # "gamma_filter":10,
    # "n_filter":5,
    # "batch_size":None,
    # "delta":4,
    # "lambda_0":10,
    # "cache_rate":1,
    # "cache_params":set(["alpha","phi","theta","acceptance","eta"]),
    # "n_iter":5000,

    "n_burn":1,
    "jobs":8,
    "verbose":True,
    "seed":42,
}

## Aggregation into Epochs
AGG_RATE = "3MO"

## Script Meta Parameters
NUM_JOBS = 8
RANDOM_SEED = 42
SAMPLE_RATE = 0.01
N_DOC_TOPIC_PLOTS = 30
MAX_PLOT_TOPICS = 10
MAX_PLOT_TERMS = 50

#####################
### Imports
#####################

## Standard Library
import os
import sys
from glob import glob
from datetime import datetime

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import matplotlib.pyplot as plt

## Local
from smtm.model import LDA, HDP, DTM, IDTM
from smtm.util.helpers import make_directory, chunks

#####################
### Globals
#####################

## Experiment Output Directory
OUTPUT_DIR = f"./data/results/observed/{MODEL_TYPE}/"

## Model Classes
MODELS = {
    "lda":LDA,
    "hdp":HDP,
    "dtm":DTM,
    "idtm":IDTM
}

## Output Directory
_ = make_directory(OUTPUT_DIR)

#####################
### Helpers
#####################

def _parse_date_frequency(freq):
    """
    Convert str-formatted frequency into seconds. Base frequencies
    include minutes (m), hours (h), days (d), weeks (w),
    months (mo), and years (y).
    Args:
        freq (str): "{int}{base_frequency}"
    
    Returns:
        period (int): Time in seconds associated with frequency
    """
    ## Frequencies in terms of seconds
    base_freqs = {
        "m":60,
        "h":60 * 60,
        "d":60 * 60 * 24,
        "w":60 * 60 * 24 * 7,
        "mo":60 * 60 * 24 * 31,
        "y":60 * 60 * 24 *  365
    }
    ## Parse String
    freq = freq.lower()
    freq_ind = 0
    while freq_ind < len(freq) - 1 and freq[freq_ind].isdigit():
        freq_ind += 1
    mult = 1
    if freq_ind > 0:
        mult = int(freq[:freq_ind])
    base_freq = freq[freq_ind:]
    if base_freq not in base_freqs:
        raise ValueError("Could not parse frequency.")
    period = mult * base_freqs.get(base_freq)
    return period
    
def get_time_bins(sample_resolution,
                  sample_reference_point="2008-01-01"):
    """
    Get linearly-spaced time bins based on a resolution

    Args:
        sample_resolution (str): Temporal frequency for sampling (size of bins)
        sample_reference_point (str): ISO-format date representing lower bound
    
    Returns:
        sample_time_bins (list of tuple): [(lower, upper),...,] epoch times in each bin
    """
    ## Indentify Bin Parameters
    sample_period_freq = _parse_date_frequency(sample_resolution)
    sample_reference_point_int = int(pd.to_datetime(sample_reference_point).timestamp())
    ## Construct Temporal Bins
    sample_time_bins = [sample_reference_point_int]
    while sample_time_bins[-1] < int(datetime.now().timestamp()):
        sample_time_bins.append(sample_time_bins[-1] + sample_period_freq)
    sample_time_bins.append(sample_time_bins[-1] + sample_period_freq)
    sample_time_bins = [(x, y) for x, y in zip(sample_time_bins[:-1], sample_time_bins[1:])]
    return sample_time_bins

## Time Bin Assigner
def batch_time_bin_assign(time_bounds,
                          time_bins):
    """
    Args:
        time_bounds (list of tuple): Lower, Upper Epoch Times
        time_bins (list of tuple): Lower, Upper Time Bin Boundaries
    """
    ## Assign Original Indice
    time_bounds_indexed = [(i, x, y) for i, (x, y) in enumerate(time_bounds)]
    ## Sort Indexed Time Bins By Lower Bound
    time_bounds_indexed = sorted(time_bounds_indexed, key=lambda x: x[1])
    ## Initialize Counters and Cache
    m = 0
    n = 0
    M = len(time_bins)
    N = len(time_bounds)
    assignments = []
    ## First Step: Assign Nulls to Bounds Before Time Bin Range
    while n < N:
        if time_bounds_indexed[n][2] < time_bins[m][0]:
            assignments.append(None)
            n += 1
        else:
            break
    ## Second Step: Assign Bins in Batches
    while n < N:
        ## Get Time Range for Data Point
        lower, upper = time_bounds_indexed[n][1], time_bounds_indexed[n][2]
        ## Check to See If Data Point Falls Outside Max Range
        if lower > time_bins[-1][0]:
            assignments.append(None)
            n += 1
            continue
        ## Increment Time Bins Until Reaching Lower Bound
        while m < M and time_bins[m][1] <= lower:
            m += 1
        ## Cache Assignment
        assignments.append(m)
        n += 1
    ## Add Assignments With Index
    assignments_indexed = [(x[0], y) for x, y in zip(time_bounds_indexed, assignments)]
    ## Sort Assignments by Original Index
    assignments_indexed = sorted(assignments_indexed, key=lambda x: x[0])
    ## Isolate Assignments
    assignments_indexed = [i[1] for i in assignments_indexed]
    return assignments_indexed

def load_document_term(output_dir):
    """

    """
    print("Loading Data From: {}".format(output_dir))
    ## Establish Filenames and Check Existence
    X_filename = f"{output_dir}/data.npz"
    post_ids_filename = f"{output_dir}/posts.txt"
    vocabulary_filename = f"{output_dir}/vocabulary.txt"
    times_filename = f"{output_dir}/times.txt"
    for f in [X_filename, post_ids_filename, vocabulary_filename, vocabulary_filename]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Could not find {f}")
    ## Load
    X = sparse.load_npz(X_filename)
    post_ids = [i.strip() for i in open(post_ids_filename,"r")]
    vocabulary = [i.strip() for i in open(vocabulary_filename,"r")]
    timestamps = [list(map(int,i.strip().split())) for i in open(times_filename,"r")]
    timestamps = np.array(timestamps)
    return X, post_ids, timestamps, vocabulary

def filter_vocabulary(X, vocabulary):
    """

    """
    df_mask = ((X!=0).sum(axis=0) >= MIN_VOCAB_DF).A[0]
    cf_mask = (X.sum(axis=0) >= MIN_VOCAB_CF).A[0]
    min_len_mask = [i for i, v in enumerate(vocabulary) if len(v) > 0 and (len(v) > 1 or not v.isalpha())] 
    mask = np.logical_and(cf_mask, df_mask).nonzero()[0]
    mask = sorted(set(mask) & set(min_len_mask))
    X = X[:,mask]
    vocabulary = [vocabulary[v] for v in mask]
    if MAX_VOCAB_SIZE is not None:
        max_size_mask = sorted(X.sum(axis=0).A[0].argsort()[-MAX_VOCAB_SIZE:])
        X = X[:,max_size_mask]
        vocabulary = [vocabulary[i] for i in max_size_mask]
    return X, vocabulary

def filter_documents(X,
                     post_ids,
                     timestamps):
    """

    """
    ## Frequency Mask
    freq_mask = (X.sum(axis=1) >= MIN_WORDS_PER_DOCUMENT).nonzero()[0]
    ## Time Mask
    min_date_dt = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
    max_date_dt = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())
    time_mask = [i for i, (l,u) in enumerate(timestamps) if l >= min_date_dt and u < max_date_dt]
    ## Sample Mask
    sample_mask = list(range(X.shape[0]))
    if SAMPLE_RATE is not None:
        sample_mask = (np.random.uniform(0,1,X.shape[0]) <= SAMPLE_RATE).nonzero()[0]
    ## Mask
    mask = sorted(set(freq_mask) & set(time_mask) & set(sample_mask))
    X = X[mask]
    post_ids = [post_ids[m] for m in mask]
    timestamps = timestamps[mask]
    return X, post_ids, timestamps

def _compute_topic_proportion(theta_df,
                              use_argmax=False):
    """

    """
    if use_argmax:
        theta_df_argmax = theta_df[theta_df.columns[1:]].idxmax(axis=1).to_frame("argmax")
        theta_df_argmax = pd.merge(theta_df[[0]],
                                   theta_df_argmax,
                                   left_index=True,
                                   right_index=True)
        theta_df_agg = pd.pivot_table(theta_df_argmax,
                                      index=0,
                                      columns="argmax",
                                      aggfunc=len).fillna(0)
    else:
        theta_df_agg = theta_df.groupby([0]).mean()
    theta_df_agg = theta_df_agg.apply(lambda i: i / sum(i), axis=1)
    return theta_df_agg

def get_topic_proportions(model,
                          timestamps_assigned,
                          time_bins_dt,
                          use_argmax=False,
                          n_bootstrap=100,
                          alpha=0.05,
                          random_seed=42):
    """

    """
    ## Set Random Seed
    _ = np.random.seed(random_seed)
    ## Generate DataFrame
    theta_df = pd.DataFrame(np.hstack([np.array(timestamps_assigned).reshape(-1,1), model.theta]))
    theta_df[0] = theta_df[0].astype(int)
    ## Baseline Statistics
    theta_agg_q = _compute_topic_proportion(theta_df, use_argmax)
    ## Bootstrap Statistics
    agg_cache = []
    for _ in tqdm(range(n_bootstrap), desc="Proportion Bootstrap", file=sys.stdout):
        theta_df_boot = theta_df.sample(n=theta_df.shape[0], replace=True)
        theta_df_boot_agg = _compute_topic_proportion(theta_df_boot, use_argmax)
        theta_df_boot_agg = theta_df_boot_agg.reindex(theta_agg_q.index)
        agg_cache.append((theta_agg_q - theta_df_boot_agg).values)
    agg_cache = np.stack(agg_cache)
    ## Summarize
    q_l, q_u = np.nanpercentile(agg_cache, q=[alpha/2 * 100, 100 - alpha/2 * 100], axis=0)
    q_l = theta_agg_q + pd.DataFrame(q_l, index=theta_agg_q.index, columns=theta_agg_q.columns)
    q_u = theta_agg_q + pd.DataFrame(q_u, index=theta_agg_q.index, columns=theta_agg_q.columns)
    ## Update Index
    for df in [theta_agg_q, q_l, q_u]:
        df.index = df.index.map(lambda i: time_bins_dt[i])
    return theta_agg_q, q_l, q_u

def plot_proportions_line(prop_mu,
                          prop_ci=None,
                          components=None):
    """
    Args:
        prop_mu (pandas DataFrame): Average Proportions
        prop_ci (tuple or None): Lower and Upper Proportion Bounds
        components (list or None): Specific Components to Plot
    """
    ## Component Check
    if components is None:
        components = prop_mu.columns.tolist()
    if prop_ci is not None:
        assert len(prop_ci) == 2
        prop_low, prop_high = prop_ci
        prop_low_diff = prop_mu - prop_low
        prop_high_diff = prop_high - prop_mu
    ## Generate Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    for i, k in enumerate(components):
        if prop_ci is None:
            ax.plot(prop_mu.index,
                    prop_mu[k].values,
                    alpha=0.5,
                    marker="o",
                    linewidth=3,
                    color=f"C{i}",
                    label=k)
        else:
            ax.errorbar(prop_mu.index,
                        prop_mu[k].values,
                        yerr=np.vstack([prop_low_diff[k].values, prop_high_diff[k].values]),
                        marker="o",
                        alpha=0.5,
                        linewidth=3,
                        color=f"C{i}",
                        capsize=5,
                        label=k)
    ax.set_xlim(prop_mu.index.min(), prop_mu.index.max())
    ax.set_ylim(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Proportion", fontweight="bold", fontsize=14)
    ax.set_xlabel("Date", fontweight="bold", fontsize=14)
    if len(components) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax

def plot_proportions_stacked(prop_mu,
                             components=None):
    """

    """
    ## Check Components
    if components is None:
        components = prop_mu.columns.tolist()
    ## Generate Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    lower = np.zeros(prop_mu.shape[0])
    for k in components:
        ax.fill_between(prop_mu.index,
                        lower,
                        lower + prop_mu[k].values,
                        alpha=0.5,
                        label=k)
        lower += prop_mu[k].values
    ax.fill_between(prop_mu.index,
                    lower,
                    np.ones(prop_mu.shape[0]),
                    alpha=0.5,
                    label="Other")
    ax.set_xlim(prop_mu.index.min(), prop_mu.index.max())
    ax.set_ylim(0, round(max(lower), 2) + 0.01)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Proportion", fontweight="bold", fontsize=14)
    ax.set_xlabel("Date", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=6)
    fig.tight_layout()
    fig.autofmt_xdate()
    return fig, ax

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
    n_plot_terms = min(MAX_PLOT_TERMS, n_plot_terms)
    n_dims = model.phi.shape[1] if is_dynamic else model.phi.shape[0]
    ## Get Epochs and Params
    trace_epochs = [None] if not is_dynamic else range(model.phi.shape[0])
    trace_params = model.cache_params
    ## Directory Setup Or Early Exit
    if len(trace_params) > 0:
        _ = make_directory(f"{OUTPUT_DIR}/trace/", remove_existing=True)
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
                figure[0].savefig(f"{OUTPUT_DIR}/trace/theta_{doc_id}.png", dpi=100)
                plt.close(figure[0])
    ## Acceptance Trace (iDTM only)
    if "acceptance" in trace_params and isinstance(model, IDTM):
        figure = model.plot_acceptance_trace()
        if isinstance(figure, tuple) and figure[0] is not None:
            figure[0].savefig(f"{OUTPUT_DIR}/trace/acceptance.png", dpi=100)
            plt.close(figure[0])
    ## Eta (iDTM only)
    if "eta" in trace_params and isinstance(model, IDTM):
        figure = model.plot_eta_trace()
        if isinstance(figure, tuple) and figure[0] is not None:
            figure[0].savefig(f"{OUTPUT_DIR}/trace/eta.png", dpi=100)
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
                    figure[0].savefig(f"{OUTPUT_DIR}/trace/alpha_{epoch_name}.png", dpi=100)
                    plt.close(figure[0])
            else:
                figure = model.plot_alpha_trace(epochs=[epoch])
                if isinstance(figure, tuple) and figure[0] is not None:
                    figure[0].savefig(f"{OUTPUT_DIR}/trace/alpha_{epoch_name}.png", dpi=100)
                    plt.close(figure[0])
        if "gamma" in trace_params and isinstance(model, IDTM):
            figure = model.plot_gamma_trace(epochs=[epoch])
            if isinstance(figure, tuple) and figure[0] is not None:
                figure[0].savefig(f"{OUTPUT_DIR}/trace/gamma_{epoch_name}.png", dpi=100)
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
                    figure[0].savefig(f"{OUTPUT_DIR}/trace/phi_{epoch_name}_{kdim}.png", dpi=100)
                    plt.close(figure[0])

def plot_topic_evolution(model):
    """

    """
    ## Number of Terms to Plot
    if isinstance(model, IDTM):
        n_plot_terms =  min(MAX_PLOT_TERMS, model.V)
    elif isinstance(model, DTM):
        n_plot_terms = min(MAX_PLOT_TERMS, len(model.vocabulary))
    ## Get Epochs and Params
    n_dims = model.phi.shape[1]
    topic_dims = range(n_dims)
    ## Directory
    _ = make_directory(f"{OUTPUT_DIR}plots/topic_evolution/", remove_existing=True)
    ## Generate Figures
    for kdim in tqdm(topic_dims, desc="Topic Evolution Plots", file=sys.stdout):
        figure = model.plot_topic_evolution(kdim, n_plot_terms)
        if isinstance(figure, tuple) and figure[0] is not None:
            figure[0].savefig(f"{OUTPUT_DIR}plots/topic_evolution/{kdim}.png", dpi=100)
            plt.close(figure[0])

def extract_dynamic_topic_summary(model,
                                  time_bins_dt,
                                  top_k=10):
    """

    """
    ## Format Time Bins
    time_bins_dt_iso = [t.isoformat() for t in time_bins_dt]
    ## Extract
    component_terms = []
    vocabulary = model.model.used_vocabs if isinstance(model, DTM) else model.vocabulary
    for c in range(model.phi.shape[1]):
        c_sort = model.phi[:,c,:].argsort(axis=1)[:,-top_k:][:,::-1]
        c_sort = [[vocabulary[i] for i in c] for c in c_sort]
        c_sort_overall = model.phi[:,c,:].mean(axis=0).argsort()[-top_k * 2:][::-1]
        c_sort_overall = [vocabulary[i] for i in c_sort_overall]
        component_terms.append((c_sort_overall, pd.DataFrame(c_sort, index=time_bins_dt_iso).T))
    ## Format
    output_str = ""
    for c, cdata in enumerate(component_terms):
        c_str = "~"*50 + f" Topic {c} " + "~"*50 + "\n"
        c_str += "Overall: {}\n\n".format(", ".join(cdata[0]))
        c_str += "Change Over Time:\n"
        c_str += cdata[1].to_string(index=False)
        output_str += f"\n{c_str}\n"
    return output_str

def main():
    """

    """
    ## Load Data
    X, post_ids, timestamps, vocabulary = load_document_term(f"{DATA_DIR}/{DATA_TYPE}/")
    ## Filter Vocabulary
    X, vocabulary = filter_vocabulary(X, vocabulary)
    ## Filter Empty Documents and By Time
    X, post_ids, timestamps = filter_documents(X, post_ids, timestamps)
    ## Time Bins
    time_bins = get_time_bins(AGG_RATE, MIN_DATE)
    time_bins_dt = [datetime.fromtimestamp(d[0]).date() for d in time_bins]
    timestamps_assigned = batch_time_bin_assign(timestamps, time_bins)
    ## Re-index Bins By Minimum
    tb_min = min(timestamps_assigned)
    tb_max = max(timestamps_assigned)
    timestamps_assigned = [t - tb_min for t in timestamps_assigned]
    time_bins = time_bins[tb_min:tb_max+1]
    time_bins_dt = time_bins_dt[tb_min:tb_max+1]
    ## Initialize Model and Parameters
    MODEL_PARAMS["vocabulary"] = vocabulary
    if MODEL_TYPE in set(["dtm","idtm"]):
        MODEL_PARAMS["t"] = max(timestamps_assigned) + 1
    model = MODELS.get(MODEL_TYPE)(**MODEL_PARAMS)
    ## Fit Model
    print("Training Model")
    fit_kwargs = {}
    if isinstance(model, DTM):
        fit_kwargs.update({"labels":timestamps_assigned,"labels_key":"timepoint"})
    elif isinstance(model, IDTM):
        fit_kwargs.update({"timepoints":timestamps_assigned})
    model = model.fit(X,
                      checkpoint_location=f"{OUTPUT_DIR}",
                      checkpoint_frequency=CHECKPOINT_FREQUENCY,
                      **fit_kwargs)
    ## Save Model
    _ = model.save(f"{OUTPUT_DIR}model.joblib")
    ## Save Model Summary
    if not isinstance(model, IDTM):
        _ = model.summary(topic_word_top_n=15, file=open(f"{OUTPUT_DIR}topic_model.summary.txt","w"))
    ## Topic Evolution
    if isinstance(model, DTM) or isinstance(model, IDTM):
        ## Plots
        _ = plot_topic_evolution(model)
        ## Summarization
        topic_summary = extract_dynamic_topic_summary(model, time_bins_dt, 10)
        with open(f"{OUTPUT_DIR}topic_model.dynamic.summary.txt","w") as the_file:
            the_file.write(topic_summary)
    ## Trace Plots
    _ = plot_traces(model, MODEL_TYPE, RANDOM_SEED, 5)
    ## Extract Topic Proportions over Time
    prop_mu, prop_low, prop_high = get_topic_proportions(model,
                                                         timestamps_assigned,
                                                         time_bins_dt,
                                                         n_bootstrap=100,
                                                         alpha=0.05,
                                                         use_argmax=False)
    ## Cache Proportions
    _ = prop_mu.to_csv(f"{OUTPUT_DIR}proportions.csv")
    _ = prop_low.to_csv(f"{OUTPUT_DIR}proportions_lower.csv")
    _ = prop_high.to_csv(f"{OUTPUT_DIR}proportions_upper.csv")
    ## Plot Component Change
    _ = make_directory(f"{OUTPUT_DIR}plots/topic_deltas/")
    component_groups = list(chunks(prop_mu.columns, MAX_PLOT_TOPICS))
    for g, group in tqdm(enumerate(component_groups),
                         desc="Component Change Plots",
                         file=sys.stdout,
                         total=len(component_groups)):
        fig, ax = plot_proportions_line(prop_mu,
                                        prop_ci=(prop_low, prop_high),
                                        components=group)
        fig.savefig(f"{OUTPUT_DIR}plots/topic_deltas/line_group_{g+1}.png",dpi=100)
        plt.close(fig)
    print("Script Complete!")

#####################
### Execution
#####################

if __name__ == "__main__":
    _ = main()

