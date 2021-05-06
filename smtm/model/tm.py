
"""

"""

###################
### Imports
###################

## Standard Library
import os
import sys

## External
import numpy as np
import tomotopy as tp
from scipy import sparse

## Local
from .base import TopicModel

###################
### Globals
###################


###################
### Classes
###################

class BaseTomotopy(TopicModel):

    """

    """

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

    def fit(self,
            X,
            labels=None,
            labels_key=None):
        """

        """
        ## Generate Corpora
        corpus_train, missing = self._add_documents(X, labels=labels, labels_key=labels_key)
        ## Add Corpus to Model
        _ = self.model.add_corpus(corpus_train)
        ## (Re-)Initialize Caches
        self._phi = []
        self._alpha = []
        self._theta = []
        self._eta = []
        self._ll = []
        ## Fit Model
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
        log_ll = np.array(log_ll)
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
             filename):
        """

        """
        _ = self.model.save(filename)
    
    def load(self,
             filename):
        """

        """
        if not os.path.exists(filename):
            raise ValueError(f"Model not found at path: {filename}")
        self.model = self.model.load(filename)

class LDA(BaseTomotopy):

    """

    """

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
        self.model = tp.LDAModel(**kwargs)
        self.model.burn_in = n_burn

    def __repr__(self):
        """

        """
        return "LDA()"

class HDP(BaseTomotopy):

    """

    """

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
        self.model = tp.HDPModel(**kwargs)
        self.model.burn_in = n_burn

    def __repr__(self):
        """

        """
        return "HDP()"
    
    def _get_topic_indices(self):
        """

        """
        topic_inds = [k for k in range(self.model.k) if self.model.is_live_topic(k)]
        return topic_inds

class DTM(BaseTomotopy):

    """

    """

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
        self.model = tp.DTModel(**kwargs)
        self.model.burn_in = n_burn
    
    def __repr__(self):
        """

        """
        return "DTM()"
