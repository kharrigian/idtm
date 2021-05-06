
"""

"""

#####################
### Imports
#####################

## Standard Library
import os
import sys
from datetime import datetime

## External Libraries
from tqdm import tqdm
from tomotopy.utils import Corpus
from scipy import sparse

#####################
### Class
#####################

class TopicModel(object):

    """

    """

    def __init__(self,
                 vocabulary=None,
                 n_iter=100,
                 n_burn=25,
                 cache_rate=None,
                 cache_params=set(),
                 jobs=1,
                 verbose=False):
        """

        """
        self.vocabulary=vocabulary
        self.n_iter=n_iter
        self.n_burn=n_burn
        self.cache_rate=cache_rate
        self.cache_params=cache_params
        self.jobs=jobs
        self.verbose=verbose
    
    def __repr__(self):
        """

        """
        return "TopicModel()"

    def _wrapper(self,
                 x,
                 desc="Iteration",
                 total=None,
                 position=0,
                 leave=True):
        """

        """
        if self.verbose:
            return tqdm(x, desc=desc, total=total, position=position, leave=leave, file=sys.stdout)
        return x

    def _add_documents(self,
                       X,
                       corpus=None,
                       labels=None,
                       labels_key=None):
        """

        """
        ## Check Data Type
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        ## Add Documents to Corpus
        if corpus is None:
            corpus = Corpus()
        missing = set()
        for k, x in self._wrapper(enumerate(X), total=X.shape[0], desc="Generating Corpus"):
            xn = x.nonzero()[1]
            if len(xn) == 0:
                missing.add(k)
                continue
            xc = [[i]*j for i, j in zip(xn, x[0, xn].A[0])]
            xc = [i for j in xc for i in j]
            if self.vocabulary is not None:
                xv = [self.vocabulary[i] for i in xc]
            else:
                xv = list(map(str, xc))
            xv_args = {}
            if labels is not None:
                xv_args[labels_key] = labels[k]
            corpus.add_doc(xv, **xv_args)
        return corpus, missing
    
    def fit(self,
            X,
            labels=None):
        """

        """
        raise NotImplementedError()
    
    def infer(self,
              X,
              labels=None):
        """

        """
        raise NotImplementedError()
    
    def save(self,
             filename):
        """

        """
        raise NotImplementedError()
    
    def load(self,
             filename):
        """

        """
        raise NotImplementedError()