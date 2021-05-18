
"""
Infinite Dynamic Topic Model (iDTM)
"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
from datetime import datetime
from collections import Counter
from multiprocessing import Pool

## External Libraries
import numpy as np
from tqdm import tqdm
from scipy import stats, sparse
from scipy.special import logsumexp

## Local
from .base import TopicModel

#######################
### Globals
#######################


#######################
### Class
#######################

class IDTM(TopicModel):

    """

    """

    def __init__(self,
                 initial_k=1,
                 initial_m=3,
                 eta=.01,
                 alpha_0_a=1,
                 alpha_0_b=1,
                 gamma_0_a=1,
                 gamma_0_b=1,
                 sigma_0=10,
                 rho_0=0.01,
                 delta=4,
                 lambda_0=0.5,
                 q=5,
                 t=1,
                 binarize=False,
                 vocabulary=None,
                 n_iter=100,
                 n_burn=25,
                 cache_rate=None,
                 cache_params=set(),
                 jobs=1,
                 seed=42,
                 verbose=False):
        """

        """
        ## General Parameters
        super().__init__(vocabulary=vocabulary,
                         n_iter=n_iter,
                         n_burn=n_burn,
                         cache_rate=cache_rate,
                         cache_params=cache_params,
                         jobs=jobs,
                         verbose=verbose)
        ## Distribution Parameters
        self._initial_k = initial_k
        self._initial_m = initial_m
        self._eta = eta
        self._alpha_0 = (alpha_0_a, alpha_0_b)
        self._gamma_0 = (gamma_0_a, gamma_0_b)
        self._rho_0 = rho_0
        self._sigma_0 = sigma_0
        ## Delta-order Process Parameters
        self._delta = delta
        self._lambda_0 = lambda_0
        ## Sampler Parameters
        self._q = q
        ## Data Parameters
        self._t = t
        self._binarize=binarize
        self._seed = seed
    
    def __repr__(self):
        """

        """
        return "IDTM()"

    def _sparse_to_list(self,
                        items):
        """

        """
        ## Separate Index
        i, items = items
        ## Transform
        if self._binarize:
            x_l = list(items.nonzero()[1])
        else:
            x_l = [i for i in items.nonzero()[1] for j in range(items[0,i])]
        return i, x_l

    def _construct_item_frequency_list(self,
                                       X):
        """

        """
        ## Check Type
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        ## Make Term List
        mp = Pool(self.jobs)
        A = list(mp.imap_unordered(self._sparse_to_list, enumerate(X)))
        _ = mp.close()
        ## Sort
        A = list(map(lambda i: i[1], sorted(A, key=lambda x: x[0])))
        return A
    
    def _initialize_bookkeeping(self,
                                X,
                                timepoints):
        """

        """
        ## Vocabulary Size
        self.V = len(self.vocabulary)
        ## Initialize Random Table Assignments For Each Word [n x w_n]
        self.word2table = [[np.random.randint(self._initial_k) for _ in x] for x in X]
        ## Initialize Random Dish Assignments for Each Table [n x m_n]
        self.table2dish = [[np.random.randint(self._initial_m) for _ in range(self._initial_k)] for _ in X]
        ## Epoch Associated with Each Document
        self.rest2epoch = {t:[] for t in range(self._t)}
        for i, t in enumerate(timepoints):
            self.rest2epoch[t].append(i)
        ## Component Counts For Each Epoch [t x 1]
        self.K_t = [0 for _ in range(self._t)]
        for epoch, documents in self.rest2epoch.items():
            epoch_comp_counts = Counter()
            for d in documents:
                for comp_k in self.table2dish[d]:
                    epoch_comp_counts[comp_k] += 1
            self.K_t[epoch] = len(epoch_comp_counts)
        ## Space for Sampling
        initial_kmax = max(self.V // 2, max(self.K_t))
        initial_mmax = max(max(list(map(len,X))) // 2, self._initial_m)
        ## Table and Component Counts (m: [t x k_max], n: [n x m_max])
        self.m = np.array([[0 for _ in range(initial_kmax)] for _ in range(self._t)])
        self.n = np.array([[0 for _ in range(initial_mmax)] for _ in X])
        for epoch, documents in self.rest2epoch.items():
            for d in documents:
                for table_k in self.table2dish[d]:
                    self.m[epoch][table_k] += 1
                    self.n[d][table_k] += 1
        ## Initialize Components [t x k_max x V]
        self.phi = [None for _ in range(self._t)]
        self.phi[0] = stats.dirichlet([self._eta]*self.V).rvs(initial_kmax)
        for epoch in range(1, self._t):
            self.phi[epoch] = np.zeros(self.phi[0].shape)
            for k, comp in enumerate(self.phi[epoch-1]):
                self.phi[epoch][k] = stats.dirichlet(comp + self._eta).rvs()[0]
        self.phi = np.stack(self.phi)
        ## Active Components in Each Epoch [t x k_t]
        self._live_k = [list(range(k)) for k in self.K_t]
        ## Initialize Word Counts ([t x k_max x V] frequency of each word for each k for each t)
        self.Z = np.zeros_like(self.phi)
        for epoch, documents in self.rest2epoch.items():
            for d in documents:
                for w_d, t_d in zip(X[d], self.word2table[d]):
                    k_d = self.table2dish[d][t_d]
                    self.Z[epoch,k_d,w_d] += 1

    def fit(self,
            X,
            timepoints):
        """

        """
        ## Vocabulary Check
        if self.vocabulary is None:
            self.vocabulary = list(range(X.shape[1]))
        ## Convert X to a List
        X = self._construct_item_frequency_list(X)
        ## Initialize Bookkeeping
        _ = self._initialize_bookkeeping(X, timepoints)
        ## Inference Loop
        for iteration in range(self.n_iter):

            ##### Sample a Topic for Each Table
            m_kt_prime = np.zeros(self.m.shape)
            for epoch in range(self._t):
                
                ## Compute m_kt'
                if epoch == 0:
                    pass
                elif (epoch + 1) < self._delta:
                    m_kt_prime[epoch] = (m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0)
                elif (epoch + 1) >= self._delta:
                    m_kt_prime[epoch] = (m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0) - \
                                         self.m[epoch-(self._delta+1)] * np.exp(-(self._delta + 1) / self._lambda_0)


