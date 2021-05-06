
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

## Topic Model Parameters
MODEL_TYPE = "lda"
MODEL_PARAMS = {
    "alpha":0.01,
    "eta":0.01,
    "k":100,
    "n_iter":1000,
    "n_burn":100,
    "cache_rate":10,
    "cache_params":set(["phi","alpha"]),
    "jobs":8,
    "verbose":True,
    "seed":42,
}

## Aggregation into Epochs
AGG_RATE = "1W"

## Script Meta Parameters
NUM_JOBS = 8
RANDOM_SEED = 42
SAMPLE_RATE = None

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
from scipy import sparse
import matplotlib.pyplot as plt

## Local
from smtm.model import LDA, HDP, DTM

#####################
### Globals
#####################

## Model Classes
MODELS = {
    "lda":LDA,
    "hdp":HDP,
    "dtm":DTM
}

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
    mask = np.logical_and(cf_mask, df_mask).nonzero()[0]
    X = X[:,mask]
    vocabulary = [vocabulary[v] for v in mask]
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
    ## Mask
    mask = sorted(set(freq_mask) & set(time_mask))
    X = X[mask]
    post_ids = [post_ids[m] for m in mask]
    timestamps = timestamps[mask]
    return X, post_ids, timestamps

def plot_trace(model,
               parameter,
               dimension=None,
               top_k=50):
    """

    """
    ## Extract
    param_cache = getattr(model, f"_{parameter}", None)
    if param_cache is None:
        return None
    ## Prepare for Plotting
    epochs = [p[0] for p in param_cache]
    data = np.stack([p[1] for p in param_cache])
    if dimension is not None and len(data.shape) > 2:
        data = data[:,dimension,:]
    if top_k is not None and data.shape[1] > top_k:
        top_k_i = np.argsort(data.mean(axis=0))[-top_k:]
        data = data[:, top_k_i]
    ## Plot Figure
    fig, ax = plt.subplots(figsize=(10,5.8))
    for d, l in enumerate(data.T):
        ax.plot(epochs, l, alpha=0.4)
    if np.min(data) > 0:
        ax.set_ylim(0)
    ax.set_xlim(0, max(epochs))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
    ax.set_ylabel("Parameter", fontweight="bold", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    return fig, ax

def get_topic_proportions(model,
                          timestamps_assigned,
                          time_bins_dt,
                          use_argmax=False,):
    """

    """
    ## Generate DataFrame
    theta_df = pd.DataFrame(np.hstack([np.array(timestamps_assigned).reshape(-1,1), model.theta]))
    theta_df[0] = theta_df[0].astype(int)
    ## Aggregate over Time
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
    theta_df_agg.index = theta_df_agg.index.map(lambda i: time_bins_dt[i])
    return theta_df_agg

def plot_proportions_line(theta_df_agg,
                          components,
                          top_k=25):
    """

    """
    ## Generate Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    for i, k in enumerate(components):
        ax.plot(theta_df_agg.index,
                theta_df_agg[k].values,
                alpha=0.5,
                linewidth=3,
                color=f"C{i}",
                label=k)
    ax.set_xlim(theta_df_agg.index.min(), theta_df_agg.index.max())
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

def plot_proportions_stacked(theta_df_agg,
                             top_k=25):
    """

    """
    ## Get Components to Plot
    top_k_components = sorted(theta_df_agg.sum(axis=0).nlargest(top_k).index.tolist())
    ## Generate Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    lower = np.zeros(theta_df_agg.shape[0])
    for k in top_k_components:
        ax.fill_between(theta_df_agg.index,
                        lower,
                        lower + theta_df_agg[k].values,
                        alpha=0.5,
                        label=k)
        lower += theta_df_agg[k].values
    ax.fill_between(theta_df_agg.index,
                    lower,
                    np.ones(theta_df_agg.shape[0]),
                    alpha=0.5,
                    label="Other")
    ax.set_xlim(theta_df_agg.index.min(), theta_df_agg.index.max())
    ax.set_ylim(0, round(max(lower), 2) + 0.01)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Proportion", fontweight="bold", fontsize=14)
    ax.set_xlabel("Date", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=6)
    fig.tight_layout()
    fig.autofmt_xdate()
    return fig, ax

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
    ## Initialize Model
    MODEL_PARAMS["vocabulary"] = vocabulary
    model = MODELS.get(MODEL_TYPE)(**MODEL_PARAMS)
    ## Fit Model
    fit_kwargs = {}
    if MODEL_TYPE in set(["dtm","idtm"]):
        fit_kwargs.update({"label":timestamps_assigned,"labels_key":"timepoint"})
    model = model.fit(X, **fit_kwargs)
    ## Extract Inferences
