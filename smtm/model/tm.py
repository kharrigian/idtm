
"""

"""

###################
### Imports
###################

## Standard Library
import os
import sys
from copy import deepcopy

## External
import joblib
import numpy as np
import pandas as pd
import tomotopy as tp
from scipy import sparse
import matplotlib.pyplot as plt

## Local
from .base import TopicModel
from ..util.helpers import make_directory

###################
### Globals
###################


###################
### Classes
###################

class BaseTomotopy(TopicModel):

    """

    """

    ## Model Loader
    _model_loader = None

    def __init__(self,
                 vocabulary=None,
                 n_iter=100,
                 n_burn=25,
                 n_sample=100,
                 cache_rate=None,
                 cache_params=set(),
                 jobs=1,
                 verbose=None):
        """

        """
        super().__init__(vocabulary=vocabulary,
                         n_iter=n_iter,
                         n_burn=n_burn,
                         cache_rate=cache_rate,
                         cache_params=cache_params,
                         jobs=jobs,
                         verbose=verbose)
        self.model=None
        self.n_sample=n_sample
    
    def __repr__(self):
        """

        """
        return "BaseTomotopy()"

    def _get_topic_indices(self):
        """

        """
        return list(range(self.model.k))

    def _plot_trace(self,
                    parameter,
                    dimension=None,
                    epoch=None,
                    top_k=50,
                    fig=None,
                    ax=None):
        """

        """
        ## Argument Validation
        if isinstance(self.model, tp.DTModel) and parameter in ["alpha","phi"] and epoch is None:
            raise ValueError("Must supply an epoch for this type of trace plot.")
        if isinstance(self.model, tp.DTModel) and parameter == "alpha":
            dimension = epoch
        ## Extract
        param_cache = getattr(self, f"_{parameter}", None)
        if param_cache is None or len(param_cache) == 0:
            return None, None
        ## Prepare for Plotting
        mcmc_epochs = [p[0] for p in param_cache]
        if isinstance(self.model, tp.DTModel) and parameter == "phi":
            data = np.stack([p[1][epoch,:,:] for p in param_cache])
        elif isinstance(self.model, tp.HDPModel):
            if parameter == "phi":
                max_k = max([p[1].shape[0] for p in param_cache])
                data = np.stack([np.vstack([p[1], np.zeros((max_k - p[1].shape[0], p[1].shape[1]))])
                                for p in param_cache])        
            elif parameter == "theta":
                max_k = max([p[1].shape[1] for p in param_cache])
                data = np.vstack([np.hstack([p[1][dimension], np.zeros(max_k - p[1].shape[1])])
                                 for p in param_cache])
            elif parameter == "alpha":
                data = np.array([p[1] for p in param_cache])
        else:
            data = np.stack([p[1] for p in param_cache])
        if dimension is not None and len(data.shape) > 2:
            data = data[:,dimension,:]
        if top_k is not None:
            if len(data.shape) > 1 and data.shape[1] > top_k:
                top_k_i = np.argsort(data.mean(axis=0))[-top_k:]
                data = data[:, top_k_i]
        ## Plot Figure
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.8))
        if len(data.shape) == 1:
            ax.plot(mcmc_epochs, data)
        else:
            for d, l in enumerate(data.T):
                ax.plot(mcmc_epochs, l, alpha=0.4)
        if np.min(data) > 0:
            ax.set_ylim(0)
        ax.set_xlim(0, max(mcmc_epochs))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
        ax.set_ylabel("Sample", fontweight="bold", fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        return fig, ax

    def _plot_topic_terms(self,
                          topic_id,
                          epoch=None,
                          top_k=None,
                          alpha=0.05,
                          fig=None,
                          ax=None):
        """

        """
        ## Validation
        if isinstance(self.model, tp.DTModel) and epoch is None:
            raise ValueError("Must supply an epoch for this type of topic distribution plot.")
        ## Extract
        param_cache = getattr(self, "_phi", None)
        if param_cache is None or len(param_cache) == 0:
            return None, None
        ## Prepare for Plotting
        if isinstance(self.model, tp.DTModel):
            data = np.stack([p[1][epoch,:,:] for p in param_cache])
        elif isinstance(self.model, tp.HDPModel):
            max_k = max([p[1].shape[0] for p in param_cache])
            data = np.stack([np.vstack([p[1], np.zeros((max_k - p[1].shape[0], p[1].shape[1]))])
                            for p in param_cache])
        else:
            data = np.stack([p[1] for p in param_cache])
        if len(data.shape) > 2:
            data = data[:,topic_id,:]
        terms = self.model.used_vocabs
        if top_k is not None and data.shape[1] > top_k:
            top_k_i = np.argsort(data.mean(axis=0))[-top_k:]
            data = data[:, top_k_i]
            terms = [terms[i] for i in top_k_i]
        ## Summarize Bounds
        data = pd.DataFrame(index=terms,
                            data=np.percentile(data, axis=0, q=[alpha/2 * 100,50,100-alpha/2 * 100]).T,
                            columns=["lower","median","upper"])
        data = data.sort_values("median", ascending=True)
        index = list(range(data.shape[0]))
        ## Create Plot
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.8))
        ax.barh(index,
                left=data["lower"],
                width=data["upper"]-data["lower"],
                color="C0",
                alpha=0.3,
                label="{:.0f}% C.I.".format(100-(alpha*100)))
        ax.scatter(data["median"],
                   index,
                   color="C0",
                   alpha=0.8,
                   s=50)
        ax.set_ylim(-.5, max(index)+.5)
        ax.set_xlim(left=max(data["lower"].min() - 0.01, 0))
        ax.set_yticks(index)
        ax.legend(loc="lower right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticklabels(data.index.tolist(), fontsize=8)
        ax.set_xlabel("Loading", fontweight="bold", fontsize=14)
        ax.tick_params(axis="x", labelsize=14)
        fig.tight_layout()
        return fig, ax

    def plot_document_trace(self,
                            doc_id,
                            top_k_topics=None):
        """

        """
        fig, ax = self._plot_trace("theta",
                                   dimension=doc_id,
                                   top_k=top_k_topics)
        if fig is None:
            return None
        return fig, ax
    
    def plot_alpha_trace(self,
                         epoch=None,
                         top_k_topics=None):
        """

        """
        fig, ax = self._plot_trace("alpha",
                                   epoch=epoch,
                                   dimension=None,
                                   top_k=top_k_topics)
        if fig is None:
            return None
        return fig, ax

    def plot_topic_trace(self,
                         topic_id,
                         epoch=None,
                         top_k_terms=50,
                         alpha=0.05):
        """

        """
        ## Initialize Figure
        fig, ax = plt.subplots(1, 2, figsize=(10,5.8), sharex=False, sharey=False)
        ## Plot Shaded Bar Plot of Terms
        fig, ax[0] = self._plot_topic_terms(topic_id,
                                            epoch=epoch,
                                            top_k=top_k_terms,
                                            alpha=alpha,
                                            fig=fig,
                                            ax=ax[0])
        ## Plot Trace
        fig, ax[1] = self._plot_trace("phi",
                                      epoch=epoch,
                                      dimension=topic_id,
                                      top_k=top_k_terms,
                                      fig=fig,
                                      ax=ax[1])
        if fig is None:
            return None
        fig.tight_layout()
        return fig, ax

    def fit(self,
            X,
            labels=None,
            labels_key=None,
            checkpoint_location=None,
            checkpoint_frequency=100):
        """

        """
        ## Generate Corpora
        corpus_train, missing = self._add_documents(X, labels=labels, labels_key=labels_key)
        ## Initialize Model
        self.model = self.model(corpus=corpus_train, **self._kwargs)
        ## (Re-)Initialize Caches
        self._phi = []
        self._alpha = []
        self._theta = []
        self._eta = []
        self._ll = []
        ## Fit Model
        self._current_iteration = 0
        for iteration in self._wrapper(range(self.n_iter), total=self.n_iter, desc="MCMC Iteration"):
            self.model.train(1, workers=self.jobs)
            self._ll.append((iteration, self.model.ll_per_word))
            if self.cache_rate is not None and (iteration + 1) % self.cache_rate == 0:
                if "phi" in self.cache_params:
                    if isinstance(self.model, tp.DTModel):
                        phi_iter = np.stack([np.vstack([self.model.get_topic_word_dist(topic,timepoint) for topic in range(self.model.k)]) for timepoint in range(self.model.num_timepoints)])
                    else:
                        phi_iter = np.vstack([self.model.get_topic_word_dist(topic) for topic in range(self.model.k)])
                    self._phi.append((iteration, phi_iter))
                if "theta" in self.cache_params:
                    theta_iter = np.vstack([doc.get_topic_dist() for doc in self.model.docs])
                    self._theta.append((iteration, theta_iter))
                if "alpha" in self.cache_params:
                    self._alpha.append((iteration, self.model.alpha))
            if checkpoint_location is not None and (self._current_iteration + 1) % checkpoint_frequency == 0:
                _ = make_directory(checkpoint_location, remove_existing=False)
                _ = self.save(f"{checkpoint_location}model.joblib")
            self._current_iteration += 1
        ## Cache Final Parameters
        if isinstance(self.model, tp.DTModel):
            self.phi = np.stack([np.vstack([self.model.get_topic_word_dist(topic,timepoint) for topic in range(self.model.k)]) for timepoint in range(self.model.num_timepoints)])
        else:
            self.phi = np.vstack([self.model.get_topic_word_dist(topic) for topic in range(self.model.k)])
        self.alpha = self.model.alpha
        self.theta, self.ll = self._infer(X, corpus_train, missing, True)
        return self

    def _infer(self,
               X,
               corpus_infer,
               missing,
               together=False):
        """

        """
        ## Run Inference
        result, result_ll = self.model.infer(corpus_infer,
                                             iter=self.n_sample,
                                             workers=self.jobs,
                                             together=together)
        ## Extract Result
        current_full_ind = 0
        theta = []
        log_ll = []
        for ind in range(X.shape[0]):
            if ind in missing:
                theta.append(np.zeros(self.model.k))
                log_ll.append(-np.inf)
            else:
                theta.append(result[current_full_ind].get_topic_dist())
                if together:
                    log_ll.append(result_ll / len(corpus_infer))
                else:
                    log_ll.append(result_ll[current_full_ind])
                current_full_ind += 1
        theta = np.vstack(theta)
        log_ll = np.hstack(log_ll)
        theta = theta[:,self._get_topic_indices()]
        return theta, log_ll

    def infer(self,
              X,
              labels=None,
              labels_key=None,
              together=False):
        """

        """
        ## Generate Corpus
        corpus_infer, missing = self._add_documents(X, labels=labels, labels_key=labels_key)
        ## Get Inferences
        theta, log_ll = self._infer(X, corpus_infer, missing, together)
        return theta, log_ll
    
    def summary(self,
                **kwargs):
        """

        """
        if self.model is None:
            raise ValueError()
        _ = self.model.summary(**kwargs)
    
    def save(self,
             filename,
             **kwargs):
        """

        """
        print(f"Saving Model to `{filename}`")
        ## Update filename
        if not filename.endswith(".joblib"):
            filename = f"{filename}.joblib"
        ## Save Tomotopy Model
        _ = self.model.save(filename.replace(".joblib",".model"))
        ## Save Remaining Class
        self.model = None
        _ = joblib.dump(self, filename, **kwargs)
        ## Reload the Model
        self.model = self._model_loader(filename.replace(".joblib",".model"))

    @staticmethod
    def load(filename,
             _model_loader):
        """

        """
        print(f"Loading Model `{filename}`...")
        if not os.path.exists(filename):
            raise ValueError(f"Model not found at path: {filename}")
        if not os.path.exists(filename.replace(".joblib",".model")):
            raise ValueError(f"Did not find tomotopy model: {filename}")
        model = joblib.load(filename)
        model.model = _model_loader(filename.replace(".joblib",".model"))
        return model

class LDA(BaseTomotopy):

    """

    """

    ## Class-specific Model Loader
    _model_loader = tp.LDAModel.load

    def __init__(self,
                 vocabulary=None,
                 n_iter=100,
                 n_burn=25,
                 n_sample=100,
                 cache_rate=None,
                 cache_params=set(),
                 jobs=1,
                 verbose=None,
                 **kwargs):
        """

        """
        super().__init__(vocabulary=vocabulary,
                         n_iter=n_iter,
                         n_burn=n_burn,
                         n_sample=n_sample,
                         cache_rate=cache_rate,
                         cache_params=cache_params,
                         jobs=jobs,
                         verbose=verbose)
        self.model = tp.LDAModel
        self._kwargs = kwargs

    def __repr__(self):
        """

        """
        return "LDA()"

    @staticmethod
    def load(filename):
        """

        """
        return BaseTomotopy.load(filename, tp.LDAModel.load)

class HDP(BaseTomotopy):

    """

    """

    ## Class-specific Model Loader
    _model_loader = tp.HDPModel.load

    def __init__(self,
                 vocabulary=None,
                 n_iter=100,
                 n_burn=25,
                 n_sample=100,
                 cache_rate=None,
                 cache_params=set(),
                 jobs=1,
                 verbose=None,
                 threshold=0.01,
                 **kwargs):
        """

        """
        super().__init__(vocabulary=vocabulary,
                         n_iter=n_iter,
                         n_burn=n_burn,
                         n_sample=n_sample,
                         cache_rate=cache_rate,
                         cache_params=cache_params,                         
                         jobs=jobs,
                         verbose=verbose)
        self.model = tp.HDPModel
        self._kwargs = kwargs
        self._threshold = threshold

    def __repr__(self):
        """

        """
        return "HDP()"
    
    def _get_topic_indices(self):
        """

        """
        topic_props = self.model.get_count_by_topics() / self.model.get_count_by_topics().sum()
        topic_inds = (topic_props >= self._threshold).nonzero()[0]
        return topic_inds
    
    @staticmethod
    def load(filename):
        """

        """
        return BaseTomotopy.load(filename, tp.HDPModel.load)

class DTM(BaseTomotopy):

    """

    """

    ## Class-specific Model Loader
    _model_loader = tp.DTModel.load

    def __init__(self,
                 vocabulary=None,
                 n_iter=100,
                 n_burn=25,
                 n_sample=100,
                 cache_rate=None,
                 cache_params=set(),
                 jobs=1,
                 verbose=None,
                 **kwargs):
        """

        """
        super().__init__(vocabulary=vocabulary,
                         n_iter=n_iter,
                         n_burn=n_burn,
                         n_sample=n_sample,
                         cache_rate=cache_rate,
                         cache_params=cache_params,
                         jobs=jobs,
                         verbose=verbose)
        self.model = tp.DTModel
        self._kwargs = kwargs
    
    def plot_topic_evolution(self,
                             topic_id,
                             top_k_terms=None,
                             top_k_type=np.nanmean):
        """

        """
        ## Check Argument
        if topic_id >= self.phi.shape[1]:
            raise ValueError("Topic ID out of range")
        ## Get Data
        data = self.phi[:,topic_id,:]
        terms = self.model.used_vocabs
        if top_k_terms is not None:
            top_k_i = top_k_type(data, axis=0).argsort()[-top_k_terms:]
            data = data[:,top_k_i]
            terms = [terms[i] for i in top_k_i]
        ## Plot Evolution
        fig, ax = plt.subplots(figsize=(10,5.8))
        m = ax.imshow(data.T,
                      cmap=plt.cm.Reds,
                      interpolation="nearest",
                      aspect="auto",
                      alpha=0.5)
        ax.set_xlabel("Epoch", fontweight="bold", fontsize=14)
        ax.set_ylabel("Term", fontweight="bold", fontsize=14)
        cbar = fig.colorbar(m)
        cbar.set_label("Parameter", fontweight="bold", fontsize=14)
        ax.set_title("Topic {}".format(topic_id), fontsize=16, fontweight="bold", loc="left")
        if top_k_terms is not None:
            ax.set_yticks(list(range(len(terms))))
            ax.set_yticklabels(terms, fontsize=8 - (top_k_terms // 100))
        ax.tick_params(axis="x", labelsize=12)
        cbar.ax.tick_params(labelsize=12)
        fig.tight_layout()
        return fig, ax
    
    def __repr__(self):
        """

        """
        return "DTM()"

    @staticmethod
    def load(filename):
        """

        """
        return BaseTomotopy.load(filename, tp.DTModel.load)
