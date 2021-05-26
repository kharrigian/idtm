
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
import pandas as pd
from tqdm import tqdm
from scipy import stats, sparse
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from scipy.special import gammaln, xlogy

## Local
from .base import TopicModel
from .helpers import (sample_categorical,
                      sample_multinomial,
                      logistic_transform)
from ..util.helpers import chunks, make_directory

#######################
### Globals
#######################


#######################
### Class
#######################

class GaussianProposal(object):

    def __init__(self,
                 q_var=1,
                 q_weight=0.5):
        """

        """
        self.q_var = q_var
        self.q_weight = q_weight
        self.q = []
    
    def generate(self,
                 phi):
        """

        """
        ## Reset Proposal Params
        self.q = []
        ## Convolve Sampled and Current Previous Values
        phi_star = np.zeros_like(phi)
        phi_star[0] = stats.multivariate_normal(phi[0], self.q_var).rvs()
        self.q.append((phi[0], self.q_var))
        for t in range(1, phi.shape[0]):
            q_mu = self.q_weight * phi_star[t-1] + (1 - self.q_weight) * phi[t-1]
            self.q.append((q_mu, self.q_var))
            phi_star[t] = stats.multivariate_normal(q_mu, self.q_var).rvs()
        return phi_star

    def score(self,
              phi):
        """

        """
        assert phi.shape[0] == len(self.q)
        ll = 0
        for p, q in zip(phi, self.q):
            ll += stats.multivariate_normal(q[0], q[1]).logpdf(p)
        return ll

class IDTM(TopicModel):

    """

    """

    def __init__(self,
                 initial_k=1, ## Initial number of components available
                 initial_m=3, ## Initial number of tables available per restaurant
                 alpha_0_a=1, ## New table concentration prior
                 alpha_0_b=1, ## New table concentration prior
                 gamma_0_a=1, ## New component concentration prior
                 gamma_0_b=1, ## New component concentration prior
                 sigma_0=10, ## Base variance
                 rho_0=0.01, ## Transition variance
                 delta=4, ## Moving window
                 lambda_0=0.5, ## Decay rate
                 q=5, ## Number of auxiliary samples
                 t=1, ## Time periods
                 q_dim=1, ## HMM number of states
                 q_var=1, ## HMM variance prior
                 q_weight=0.5, ## Gaussian Proposal Weights
                 q_type="hmm",
                 alpha_filter=1, ## Fixed alpha during filtering
                 gamma_filter=1, ## Fixed gamma during filtering
                 n_filter=10, ## Number of filtering iterations
                 threshold=0.01, ## Minimum threshold for topic acceptance
                 k_filter_frequency=None, ## Frequency to filtering topics based on threshold
                 batch_size=None, ## Batch size
                 binarize=False, ## Whether to binarize the document term matrix
                 vocabulary=None, ## Vocabulary
                 n_iter=100, ## Number of training iterations
                 n_burn=25, ## Burn in
                 cache_rate=None, ## How often to cache
                 cache_params=set(), ## Which params to cache
                 jobs=1, ## Number of multiprocessing jobs
                 seed=42, ## Random Seed
                 verbose=False): ## Verbosity
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
        ## Initialization Parameters
        self._initial_k = initial_k
        self._initial_m = initial_m
        ## Priors
        self._alpha_0 = (alpha_0_a, alpha_0_b)
        self._gamma_0 = (gamma_0_a, gamma_0_b)
        self._rho_0 = rho_0
        self._sigma_0 = sigma_0
        ## Proposal Params
        self._q_dim = q_dim
        self._q_var = q_var
        self._q_type = q_type
        self._q_weight = q_weight
        ## Training Params
        self._alpha_filter = alpha_filter
        self._gamma_filter = gamma_filter
        self._n_filter = n_filter
        self._threshold = threshold
        self._batch_size = batch_size
        if self._threshold is None:
            self._threshold = 0
        self._k_filter_freq = k_filter_frequency
        ## Delta-order Process Parameters
        self._delta = delta
        self._lambda_0 = lambda_0
        ## Sampler Parameters
        self._q = q
        ## Data Parameters
        self._t = t
        self._binarize = binarize
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
        A = list(tqdm(mp.imap_unordered(self._sparse_to_list, enumerate(X)),
                 desc="Formatting Data Matrix",
                 file=sys.stdout,
                 total=X.shape[0]))
        _ = mp.close()
        ## Sort
        A = list(map(lambda i: i[1], sorted(A, key=lambda x: x[0])))
        return A
    
    def _normalize(self,
                   a,
                   **kwargs):
        """

        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        asum = a.sum(**kwargs)
        if isinstance(asum, np.ndarray):
            a_norm = np.divide(a,
                               asum,
                               out=np.zeros_like(a),
                               where=asum>0)
        else:
            a_norm = a / asum if asum != 0 else 0
        return a_norm

    def _initialize_bookkeeping(self,
                                X,
                                timepoints):
        """

        """
        print("Initializing Bookkeeping...")
        ## Vocabulary Size
        self.V = len(self.vocabulary)
        ## Initialize Random Table Assignments For Each Word [n x w_n]
        self.word2table = [list(np.random.choice(self._initial_m, size=len(x))) for x in X]
        ## Initialize Random Dish Assignments for Each Table [n x m_n]
        self.table2dish = [list(np.random.randint(self._initial_k, size=self._initial_m)) for _ in X]
        ## Epoch Associated with Each Document
        self.rest2epoch = [[] for t in range(self._t)]
        for i, t in enumerate(timepoints):
            self.rest2epoch[t].append(i)
        ## Space for Sampling
        initial_kmax = max(self._initial_k + self._initial_k // 4, 2)
        initial_mmax = max(max(max(list(map(len,X))) // 2, self._initial_m), 2)
        ## Table and Component Counts (m: [t x k_max], n: [n x m_max])
        print("Initializing Counts")
        self.m = np.zeros((self._t, initial_kmax), dtype=np.int)
        self.m_kt_prime = np.zeros(self.m.shape)
        self.n = np.zeros((len(X), initial_mmax), dtype=np.int)
        for epoch, documents in enumerate(self.rest2epoch):
            for d in documents:
                dtables = set(self.word2table[d])
                for j, dish_k in enumerate(self.table2dish[d]):
                    if j in dtables:
                        self.m[epoch][dish_k] += 1
                for table_k in self.word2table[d]:
                    self.n[d][table_k] += 1
        ## Component Lifespans
        self.K_life = np.zeros((initial_kmax, 2), dtype=np.int)
        for k in range(self.K_life.shape[0]):
            if len((self.m[:,k] != 0).nonzero()[0]) == 0:
                continue
            self.K_life[k] = [(self.m[:,k] != 0).nonzero()[0].min(), (self.m[:,k] != 0).nonzero()[0].max()]
        ## Initialize Base Distribution
        print("Initializing Components")
        self._H = stats.multivariate_normal(np.zeros(self.V), self._sigma_0)
        ## Initialize Word Counts 
        ## Z: [t x k_max x V] frequency of each word for each k for each t
        ## v: [n x m_max x V]: frequency of each word in each table for each document
        self.Z = np.zeros((self._t, initial_kmax, self.V), dtype=np.int)
        self.v = np.zeros((len(X), initial_mmax, self.V), dtype=np.int)
        for epoch, documents in enumerate(self.rest2epoch):
            for d in documents:
                for w_d, t_d in zip(X[d], self.word2table[d]):
                    k_d = self.table2dish[d][t_d]
                    self.Z[epoch,k_d,w_d] += 1
                    self.v[d, t_d, w_d] += 1
        ## Initialize Components [t x k_max x V]
        self.phi = np.zeros(self.Z.shape)
        for e, Ze in enumerate(self.Z):
            self.phi[e] = np.divide((Ze - Ze.mean(axis=1,keepdims=True)).astype(float),
                                    Ze.std(axis=1,keepdims=True).astype(float),
                                    where=Ze.std(axis=1,keepdims=True)>0,
                                    out=np.zeros_like(Ze).astype(float))
        self.phi_T = logistic_transform(self.phi, axis=2, keepdims=True)
        ## Initialize Concetration Parameters
        if self._n_filter > 0:
            self.gamma = self._gamma_filter * np.ones(self._t)
            self.alpha = self._alpha_filter * np.ones(self._t)
        else:
            self.gamma = stats.gamma(self._gamma_0[0],scale=1/self._gamma_0[1]).rvs(self._t)
            self.alpha = stats.gamma(self._alpha_0[0],scale=1/self._alpha_0[1]).rvs(self._t)
        ## Initialize Auxiliary Variable Sampler 
        self._phi_aux = self._H.rvs(self._q)
        if self._q == 1:
            self._phi_aux = self._phi_aux.reshape(1,-1)
        self._phi_aux_T = logistic_transform(self._phi_aux, axis=1, keepdims=True)
        ## Clean Up
        _ = self._clean_up()
        # Update Initialize Component Tracking
        self._component_map = [{i:None for i in range(self.m.shape[1])}]

    def _polynomial_expand(self,
                           a,
                           c):
        """
        [a]^c = a(a+1)(a+2)...(a+c-1)
        """
        expansion = (np.arange(0, c) + a).prod()
        return expansion
    
    def _log(self,
             x):
        """

        """
        return np.log(x, where=x>0, out=np.ones_like(x) * -np.inf)
    
    def _l1_norm(self,
                 x,
                 **kwargs):
        """

        """
        z = x.sum(**kwargs)
        norm = np.divide(x, z, out=np.zeros_like(x), where=z>0)
        return norm

    def _multinomial_pmf(self, 
                         x,
                         n,
                         p,
                         log=True):
        """
        Multinomial probability mass function.
        Parameters
        ----------
        x : array_like (d,)
            Quantiles.
        n : int
            Number of trials
        p : array_like, shape (k,d)
            Probabilities. These should sum to one. If they do not, then
            ``p[-1]`` is modified to account for the remaining probability so
            that ``sum(p) == 1``.
        Returns
        -------
        logpmf : float
            Log of the probability mass function evaluated at `x`. 
        """
        x = np.asarray(x)
        coef = gammaln(n + 1) - gammaln(x + 1.).sum(axis=-1)
        val = coef + np.sum(xlogy(x, p), axis=-1)
        # insist on that the support is a set of *integers*
        mask = np.logical_and.reduce(np.mod(x, 1) == 0, axis=-1)
        mask &= (x.sum(axis=-1) == n)
        out = np.where([mask] * p.shape[0], val, -np.inf)
        if not log:
            out = np.exp(out)
        return out

    def _metropolis_hastings(self,
                             phi_k,
                             phi_k_star,
                             v_k,
                             q=None,
                             smoothing=1e-10):
        """
        Compute Acceptance Ratio for Linear Emission Model and return
        selected sample

        Args:
            phi_k (2d-array): Current parameters
            phi_k_star (2d-array): Proposed parameters
            v_k (2d-array): Current counts
            q (stats frozen distribution): Proposal distribution, Gaussian HMM

        Returns:
            phi_k_selected (2d-array), phi_k_T_selected (2d-array), accept (bool)
        """
        ## Attribute Check
        if q is None and phi_k.shape[0] > 1:
            raise ValueError("Improper proposal distribution supplied")
        ## Check Shape
        if len(phi_k_star.shape) == 1:
            phi_k_star = phi_k_star.reshape(1,-1)
        ## Transformations
        phi_k_T = logistic_transform(phi_k, axis=1, keepdims=True)
        phi_k_star_T = logistic_transform(phi_k_star, axis=1, keepdims=True)
        ## Smooth to Avoid Underflow Errors
        phi_k_smooth = (phi_k_T + smoothing) / (phi_k_T + smoothing).sum(axis=1,keepdims=True)
        phi_k_star_smooth = (phi_k_star_T + smoothing) / (phi_k_star_T + smoothing).sum(axis=1,keepdims=True)
        ## Compute Initial State Probability
        h_ll = self._H.logpdf(phi_k[0])
        h_star_ll = self._H.logpdf(phi_k_star[0])
        ## Model Probabilities
        q_ll = 0 if phi_k.shape[0] == 1 else q(phi_k)
        q_star_ll = 0 if phi_k.shape[0] == 1 else q(phi_k_star)
        ## Compute Transition Probabilities
        emission_ll = []
        emission_star_ll = []
        transition_ll = []
        transition_star_ll = []
        n_k = v_k.sum(axis=1)
        for t, (v_kt, n_kt) in enumerate(zip(v_k,n_k)):
            ## Ignore First Step
            if t == 0:
                continue
            ## Proposal Data Likelihood
            emission_star_ll.append(self._multinomial_pmf(v_kt, n_kt, phi_k_star_smooth[[t]],log=True)[0])
            transition_star_ll.append(stats.multivariate_normal(phi_k_star[t-1],self._rho_0).logpdf(phi_k_star[t]))
            ## Existing Data Likelihood
            emission_star_ll.append(self._multinomial_pmf(v_kt, n_kt, phi_k_smooth[[t]],log=True)[0])
            transition_ll.append(stats.multivariate_normal(phi_k[t-1],self._rho_0).logpdf(phi_k[t]))
        ## Compute Ratio
        if v_k.shape[0] > 1:
            numerator = h_star_ll + logsumexp(emission_star_ll + transition_star_ll) + q_ll
            denominator = h_ll + logsumexp(emission_ll + transition_ll) + q_star_ll
        else:
            numerator = h_star_ll + q_ll
            denominator = h_ll + q_star_ll
        ratio_ll = numerator - denominator
        ratio = 1 if ratio_ll > 0 else min(1, np.exp(ratio_ll))
        ## Return Selected Sample
        if np.random.uniform() < ratio:
            return phi_k_star, 1
        else:
            return phi_k, 0
        
    def _clean_up(self):
        """

        """
        ## Get Empty and Nonempty Components
        nonempty_components = (self.m.sum(axis=0) !=0).nonzero()[0]
        ## New Component Mapping
        old2new_ind = dict(zip(nonempty_components, range(nonempty_components.shape[0])))
        ## Remove Empty Components
        self.m = self.m[:,nonempty_components]
        self.m_kt_prime = self.m_kt_prime[:,nonempty_components]
        self.phi = self.phi[:,nonempty_components,:]
        self.phi_T = self.phi_T[:,nonempty_components,:]
        self.Z = self.Z[:,nonempty_components,:]
        self.K_life = self.K_life[nonempty_components]
        ## Cycle Through Epochs
        for epoch, documents in enumerate(self.rest2epoch):
            ## Cycle through documents in the epoch
            for document in documents:
                ## Get Current Component Assignments
                dishes = self.table2dish[document]
                ## Identify Empty Tables
                current_tables = self.n[document]
                nonempty_tables = current_tables.nonzero()[0]
                empty_tables = (current_tables == 0).nonzero()[0]
                empty_slots = np.zeros(self.n.shape[1] - nonempty_tables.shape[0], dtype=np.int)
                old2new_table_ind = dict(zip(nonempty_tables, range(nonempty_tables.shape[0])))
                ## Table Count Update
                self.n[document] = np.hstack((self.n[document,nonempty_tables],empty_slots))
                self.v[document] = np.vstack([self.v[document,nonempty_tables],self.v[document,empty_tables]])
                self.word2table[document] = [old2new_table_ind[w] for w in self.word2table[document]]
                self.table2dish[document] = [old2new_ind[d] for d in [dishes[n] for n in nonempty_tables]]
        return old2new_ind

    def _train(self,
               X,
               iteration,
               batch,
               batch_n,
               is_last=False):
        """

        """
        ## Get Batch Indices
        batch = set(batch)
        ## Check for Component Filtering
        k_filter_on = False
        if (self._k_filter_freq is not None and iteration % self._k_filter_freq == 0) or (is_last and self._k_filter_freq is not None):
            mdist = self.m.sum(axis=0) / self.m.sum()
            k_filter = (mdist >= self._threshold).astype(int)
            k_filter_on = True
        ## Track Where Components Are Created
        k_created = [0,0]
        k_ind_created = set()
        ####### Step 1: Sample Topic k_tdb for table tdb
        ## Reset m'
        self.m_kt_prime = np.zeros(self.m.shape)
        ## Iterate over time periods
        for epoch, epoch_docs in tqdm(enumerate(self.rest2epoch), desc="Sampling Topic Assignments", total=len(self.rest2epoch), file=sys.stdout):
            ## Compute m_kt' for the Epoch
            if epoch == 0:
                pass
            elif (epoch + 1) < self._delta:
                self.m_kt_prime[epoch] = (self.m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0)
            elif (epoch + 1) >= self._delta:
                self.m_kt_prime[epoch] = (self.m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0) - \
                                          self.m[epoch-(self._delta+1)] * np.exp(-(self._delta + 1) / self._lambda_0)
            ## Cycle Through Documents
            for document in list(filter(lambda d: d in batch, epoch_docs)):
                for table, dish in enumerate(self.table2dish[document]):
                    ## Get Words Indices at Table
                    w_tdb = [w for w, t in zip(X[document], self.word2table[document]) if t == table]
                    n_tdb = len(w_tdb)
                    ## Remove Dish from Counts for the Epoch
                    self.m[epoch, dish] -= 1
                    self.Z[epoch, dish] -= self.v[document][table]
                    ## Identify Topics (Used in Epoch or Existing Previously)
                    k_used = set(self.m[epoch].nonzero()[0])
                    k_exists = set(self.m_kt_prime[epoch].nonzero()[0]) - k_used ## Exists but not Used
                    max_k = max(k_used | k_exists)
                    ## Get Conditional Probability of Data Given Table Component
                    v_tdk = self.v[document][table]
                    f_v_tdb_used = self._multinomial_pmf(v_tdk, n_tdb, self.phi_T[epoch], False)
                    if epoch > 0:
                        f_v_tdb_exists = self._multinomial_pmf(v_tdk, n_tdb, self.phi_T[epoch-1], False)
                    else:
                        f_v_tdb_exists = np.ones_like(f_v_tdb_used) / len(f_v_tdb_used)
                    f_v_tdb_new = self._multinomial_pmf(v_tdk, n_tdb, self._phi_aux_T, False)
                    ## Probabilities
                    p_k_used = (self.m[epoch] + self.m_kt_prime[epoch]) * f_v_tdb_used
                    p_k_exist = self.m_kt_prime[epoch] * f_v_tdb_exists
                    p_k_not_new = np.where([i in k_used for i in range(p_k_used.shape[0])],
                                            p_k_used,
                                            p_k_exist)
                    p_k_not_new[max_k:] = 0
                    p_k_new = self.gamma[epoch] / self._q * f_v_tdb_new
                    ## Component Filtering (Only Allocate to Existing)
                    if k_filter_on:
                        p_k_not_new = p_k_not_new * k_filter
                        p_k_new = p_k_new * 0
                    ## Merge Distribution and Normalize
                    p_k_all = np.hstack([p_k_not_new, p_k_new])
                    p_k_all = p_k_all / p_k_all.sum()
                    ## TODO: Figure out transition probability term for reweighting K_potential
                    ## Sample Topic
                    k_sample = sample_categorical(p_k_all)
                    ## Case 1: Sampled Topic Exists (Previously or In Epoch Already)
                    if k_sample <= max_k:
                        ## Update Counts
                        self.m[epoch,k_sample] += 1
                        self.table2dish[document][table] = k_sample
                        self.Z[epoch, k_sample] += self.v[document][table]
                        ## Update Component for Unused Component
                        if k_sample not in k_used:
                            self.phi[epoch,k_sample] = stats.multivariate_normal(self.phi[epoch-1,k_sample], self._rho_0).rvs()
                            self.phi_T[epoch,k_sample] = logistic_transform(self.phi[epoch,k_sample])
                        ## Update Lifespan
                        self.K_life[k_sample,1] = max(epoch, self.K_life[k_sample,1])
                    ## Case 2: Sampled Topic is Completely New
                    else:
                        k_created[0] += 1
                        ## Figure Out Which Auxiliary Component Was Selected
                        k_aux = k_sample - p_k_not_new.shape[0]
                        k_new_ind = max_k + 1
                        k_ind_created.add(k_new_ind)
                        ## Check To See if Expansion Need
                        if k_new_ind >= (self.m.shape[1] - 1):
                            exp_size = max(self.m.shape[1] // 2, 2)
                            self.m = np.hstack([self.m, np.zeros((self.m.shape[0], exp_size), dtype=np.int)])
                            self.m_kt_prime = np.hstack([self.m_kt_prime, np.zeros((self.m_kt_prime.shape[0], exp_size), dtype=np.int)])
                            self.Z = np.hstack([self.Z, np.zeros((self.Z.shape[0], exp_size, self.Z.shape[2]), dtype=np.int)])
                            self.phi = np.hstack([self.phi, np.zeros((self.phi.shape[0], exp_size, self.phi.shape[2]), dtype=np.int)])
                            self.phi_T = np.hstack([self.phi_T, np.zeros((self.phi_T.shape[0], exp_size, self.phi_T.shape[2]), dtype=np.int)])
                            self.K_life = np.vstack([self.K_life, np.zeros((exp_size, 2), dtype=np.int)])
                        ## Update Counts
                        self.m[epoch, k_new_ind] += 1
                        self.table2dish[document][table] = k_new_ind
                        self.Z[epoch, k_new_ind] += self.v[document][table]
                        ## Update Component
                        self.phi[epoch,k_new_ind] = self._phi_aux[k_aux]
                        self.phi_T[epoch,k_new_ind] = self._phi_aux_T[k_aux]
                        ## Sample New Auxiliary Component
                        self._phi_aux[k_aux] = self._H.rvs()
                        self._phi_aux_T[k_aux] = logistic_transform(self._phi_aux[k_aux])
                        ## Add to Live Components
                        self.K_life[k_new_ind] = [epoch, epoch]
        ####### Step 2: Sample a Table b_tdi for Each Word x_tdi
        for epoch, epoch_docs in tqdm(enumerate(self.rest2epoch), total=len(self.rest2epoch), desc="Sampling Tables", file=sys.stdout, position=0, leave=True):
            for document in list(filter(lambda d: d in batch, epoch_docs)):
                for i, (x_tdi, b_tdi) in enumerate(zip(X[document],self.word2table[document])):
                    ## Current Component
                    k_current = self.table2dish[document][b_tdi]
                    ## Remove Word From Table
                    self.n[document][b_tdi] -= 1
                    self.Z[epoch,k_current,x_tdi] -= 1
                    self.v[document,b_tdi,x_tdi] -= 1
                    self.word2table[document][i] = None
                    ## Remove Table if Now Empty
                    now_empty = self.n[document][b_tdi] == 0
                    if now_empty:
                        self.m[epoch, k_current] -= 1
                    ## Probability Of Word For Each Table
                    f_x_tdi = self.phi_T[epoch, self.table2dish[document], x_tdi]
                    p_x_tdi = self.n[document,:f_x_tdi.shape[0]] * f_x_tdi
                    ## Either Sample An Existing Table or a New Table
                    p_b_0 = np.hstack([p_x_tdi, np.r_[self.alpha[epoch]]])
                    p_b_0 = p_b_0 / p_b_0.sum()
                    b_0 = sample_categorical(p_b_0)
                    ## Case 1: Assign to an Existing Table
                    if b_0 < p_b_0.shape[0] - 1:
                        ## Table Component
                        k_sampled = self.table2dish[document][b_0]
                        ## Update Counts and Cache
                        self.n[document, b_0] += 1
                        self.Z[epoch,k_sampled,x_tdi] += 1
                        self.word2table[document][i] = b_0
                        self.v[document,b_0,x_tdi] += 1
                        self.K_life[k_sampled,1] = max(self.K_life[k_sampled,1], epoch)
                    ## Case 2: Create a New Table
                    else:
                        ## Identify Topics (Used in Epoch or Existing Previously)
                        k_used = set(self.m[epoch].nonzero()[0])
                        k_exists = set(self.m_kt_prime[epoch].nonzero()[0]) - k_used
                        max_k = max(k_used | k_exists)
                        ## Check Cache Size (See if Enough Tables)
                        if b_0 >= self.n.shape[1]:
                            exp_size = max(self.n.shape[1] // 4, 2)
                            self.n = np.hstack([self.n, np.zeros((self.n.shape[0],exp_size), dtype=np.int)])
                            self.v = np.hstack([self.v, np.zeros((self.v.shape[0],exp_size,self.v.shape[2]),dtype=np.int)])
                        ## Probability of Word For a New Table under Each Component
                        p_k_used = (self.m_kt_prime[epoch] + self.m[epoch]) * self.phi_T[epoch,:,x_tdi]
                        if epoch > 0:
                            p_k_exists = self.m_kt_prime[epoch] * self.phi_T[epoch-1,:,x_tdi]
                        else:
                            p_k_exists = np.ones_like(p_k_used) / len(p_k_used)
                        p_k_not_new = np.where([i in k_used for i in range(p_k_used.shape[0])],
                                                p_k_used,
                                                p_k_exists)
                        p_k_not_new[max_k:] = 0
                        p_k_new = self.gamma[epoch] / self._q * self._phi_aux_T[:,x_tdi]
                        if k_filter_on:
                            p_k_not_new = p_k_not_new * k_filter
                            p_k_new = p_k_new * 0
                        ## Construct Sample Probability
                        p_k_all = np.hstack([p_k_not_new, p_k_new])
                        p_k_all = p_k_all / p_k_all.sum()
                        ## Sample Component
                        k_sampled = sample_categorical(p_k_all)
                        ## Case 1: Sampled an Existing Component
                        if k_sampled <= max_k:
                            ## Update Counts
                            self.n[document, b_0] += 1
                            self.Z[epoch,k_sampled,x_tdi] += 1
                            self.v[document,b_0,x_tdi] += 1
                            self.word2table[document][i] = b_0
                            self.m[epoch,k_sampled] += 1
                            self.table2dish[document].append(k_sampled)
                            self.K_life[k_sampled,1] = max(self.K_life[k_sampled,1], epoch)
                            ## Update Component for Unused Component
                            if k_sampled not in k_used:
                                self.phi[epoch,k_sampled] = stats.multivariate_normal(self.phi[epoch-1,k_sampled], self._rho_0).rvs()
                                self.phi_T[epoch,k_sampled] = logistic_transform(self.phi[epoch,k_sampled])
                        ## Case 2: Sampled a New Component
                        else:
                            k_created[1] += 1
                            ## Find Appropriate Auxiliary Variable
                            k_aux = k_sampled - p_k_not_new.shape[0]
                            k_new_ind = max_k + 1
                            k_ind_created.add(k_new_ind)
                            ## Check Cache Size (Components)
                            if k_new_ind >= self.m.shape[1] - 1:
                                exp_size = max(self.m.shape[1] // 2, 2)
                                self.m = np.hstack([self.m, np.zeros((self.m.shape[0], exp_size), dtype=np.int)])
                                self.m_kt_prime = np.hstack([self.m_kt_prime, np.zeros((self.m_kt_prime.shape[0], exp_size), dtype=np.int)])
                                self.Z = np.hstack([self.Z, np.zeros((self.Z.shape[0], exp_size, self.Z.shape[2]), dtype=np.int)])
                                self.phi = np.hstack([self.phi, np.zeros((self.phi.shape[0], exp_size, self.phi.shape[2]), dtype=np.int)])
                                self.phi_T = np.hstack([self.phi_T, np.zeros((self.phi_T.shape[0], exp_size, self.phi_T.shape[2]), dtype=np.int)])
                                self.K_life = np.vstack([self.K_life, np.zeros((exp_size, 2), dtype=np.int)])
                            ## Update counts
                            self.m[epoch, k_new_ind] += 1
                            self.Z[epoch, k_new_ind, x_tdi] += 1
                            self.v[document,b_0,x_tdi] += 1
                            self.n[document,b_0] += 1
                            self.table2dish[document].append(k_new_ind)
                            self.word2table[document][i] = b_0
                            self.K_life[k_new_ind] = [epoch, epoch]
                            ## Update Component
                            self.phi[epoch,k_new_ind] = self._phi_aux[k_aux]
                            self.phi_T[epoch,k_new_ind] = self._phi_aux_T[k_aux]
                            ## Sample New Auxiliary Component
                            self._phi_aux[k_aux] = self._H.rvs()
                            self._phi_aux_T[k_aux] = logistic_transform(self._phi_aux[k_aux])
        ## Filtering Mode
        print("Sampling Concentration Parameters")
        if iteration < self._n_filter:
            self.alpha = self._alpha_filter * np.ones(self._t)
            self.gamma = self._gamma_filter * np.ones(self._t)
        else:
            ## Cycle Through Epochs
            for epoch, documents in enumerate(self.rest2epoch):
                ## Some Data Information
                n_J = self.n[documents].sum(axis=1) ## Number of Words Per Document
                m_J = (self.n[documents] != 0).sum(axis=1) ## Number of Tables per Document
                T = self.m[epoch].sum() ## Total Number of Tables
                K = len(self.m[epoch].nonzero()[0])
                ####### Step 3: Sample Concetration Parameter Alpha
                w_j = stats.beta(self.alpha[epoch] + 1, n_J).rvs()
                p_s_j = (n_J * (self._alpha_0[0] - np.log(w_j))) / (self._alpha_0[0] + m_J - 1 + n_J * (self._alpha_0[1] - np.log(w_j)))
                s_j = stats.bernoulli(p_s_j).rvs()
                self.alpha[epoch] = stats.gamma(self._alpha_0[0] + T - s_j.sum(), scale = 1 / (self._alpha_0[1] - np.log(w_j).sum())).rvs()
                ####### Step 4: Sample Concentration Parameter Gamma
                eta = stats.beta(self.gamma[epoch]+1, T).rvs()
                i_gamma_mix = int(np.random.uniform(0,1) > (self._gamma_0[0] + K - 1)/(self._gamma_0[0] + K - 1 + T*(self._gamma_0[1] - np.log(eta))))
                if i_gamma_mix == 0:
                    self.gamma[epoch] = stats.gamma(self._gamma_0[0] + K, scale = 1 / (self._gamma_0[1] - np.log(eta))).rvs()
                elif i_gamma_mix == 1:
                    self.gamma[epoch] = stats.gamma(self._gamma_0[0] + K - 1, scale = 1 / (self._gamma_0[1] - np.log(eta))).rvs()
        ## Remove Empty Components
        new_component_map = self._clean_up()
        ## Get Component Transition Map
        new_component_map_r = {y:x for x, y in new_component_map.items()}
        component_transition_map = {}
        for active_k, previous_k in new_component_map_r.items():
            if previous_k in k_ind_created:
                component_transition_map[active_k] = None
            else:
                component_transition_map[active_k] = previous_k
        self._component_map.append(component_transition_map)
        ####### Step 5: Sample Components phi_tk using Z
        n_accept = [0,0]
        self._acceptance = np.zeros(self.phi.shape[1])
        for k in tqdm(range(self.phi.shape[1]), desc="Sampling Components", file=sys.stdout, total=self.phi.shape[1]):
            ## Isolate Existing Variables
            phi_k = self.phi[:,k,:]
            v_k = self.Z[:,k,:]
            ## Filter by Lifespan
            born, die = self.K_life[k]
            phi_k = phi_k[born:die+1]
            v_k = v_k[born:die+1]
            ## Generate Proposal Distribution and Sample from It
            if phi_k.shape[0] == 1: ## Case 1: Only One Epoch
                qfunc = None
                phi_k_star = stats.multivariate_normal(phi_k[0], self._q_var).rvs()
            else:
                if self._q_type == "hmm":
                    try:
                        nn_phi = len(phi_k.sum(axis=1).nonzero()[0])
                        q = GaussianHMM(n_components=max(min(self._q_dim, nn_phi // 3), 1),
                                        covariance_type="diag",
                                        n_iter=10,
                                        means_prior=0,
                                        covars_prior=self._q_var,
                                        random_state=self._seed,
                                        verbose=False)
                        q = q.fit(phi_k)
                        phi_k_star, _ = q.sample(phi_k.shape[0])
                        qfunc = q.score
                    except:
                        q = GaussianProposal(self._q_var, self._q_weight)
                        phi_k_star = q.generate(phi_k)
                        qfunc = q.score
                elif self._q_type == "gaussian":
                    q = GaussianProposal(self._q_var, self._q_weight)
                    phi_k_star = q.generate(phi_k)
                    qfunc = q.score
            ## MH Step
            phi_k, accept = self._metropolis_hastings(phi_k,
                                                      phi_k_star,
                                                      v_k,
                                                      qfunc)
            n_accept[0] += accept
            n_accept[1] += 1
            self._acceptance[k] = accept
            ## Check Acceptance
            if not accept:
                continue
            ## Update Parameters Based on Outcome (With Temporal Padding if Necessary)
            if phi_k.shape[0] != self.phi.shape[0]:
                phi_k_padded = np.zeros_like(self.phi[:,k,:])
                phi_k_padded[born:die+1] = phi_k
                self.phi[:,k,:] = phi_k_padded
                self.phi_T[:,k,:] = logistic_transform(phi_k_padded, axis=1, keepdims=True)
            else:
                self.phi[:,k,:] = phi_k
                self.phi_T[:,k,:] = logistic_transform(phi_k, axis=1, keepdims=True)
        ## Update User on Training Progress
        if self.verbose:
            print("\nIteration {} -- Batch {} Complete\n".format(iteration+1,batch_n)+"~"*50)
            print("# Components:", max(list(map(max,self.table2dish))))
            print("Max # Tables:", max(list(map(max,self.word2table))))
            print("Number of New Components Per Sampling Stage:", k_created)
            print("Proposal Acceptance Rate:", n_accept[0] / n_accept[1])
            print("Alpha:", self.alpha)
            print("Gamma", self.gamma)
            print("Eta:", self.m.sum(axis=0))

    def _get_theta(self):
        """

        """
        k_active = max((self.m.sum(axis=0) > 0).nonzero()[0]) + 1
        theta = np.zeros((len(self.word2table), k_active))
        for document, table_assignments in enumerate(self.word2table):
            dish_assignments = [self.table2dish[document][b] for b in table_assignments]
            for k in dish_assignments:
                theta[document,k] += 1
        theta = self._l1_norm(theta, axis=1, keepdims=True)
        return theta

    def _get_component_path(self,
                            k):
        """

        """
        k_inds = [None for _ in self._component_map]
        k_inds[-1] = k
        i = -1
        while True and -i < len(self._component_map) :
            previous_ind = self._component_map[i].get(k_inds[i],None)
            k_inds[i-1] = previous_ind
            i -= 1
            if previous_ind is None:
                break
        k_inds = k_inds[1:]
        return k_inds

    def _get_phi_trace(self,
                       epoch,
                       k,
                       iter_min=None,
                       iter_max=None,
                       transform=True):
        """

        """
        ## Get Path
        kpath = self._get_component_path(k)
        ## Get Data
        epochs = []
        data = []
        for j, kind in enumerate(kpath):
            if kind is None:
                continue
            jepoch, jphi = self._phi_cache[j]
            if iter_min is not None and jepoch < iter_min:
                continue
            if iter_max is not None and jepoch > iter_max:
                continue
            epochs.append(jepoch)
            data.append(jphi[epoch,kind,:])
        if len(data) == 0:
            return None, None
        data = np.vstack(data)
        if transform:
            data = logistic_transform(data, axis=1, keepdims=True)
        return epochs, data

    def _plot_phi_dist(self,
                       epoch,
                       k,
                       min_iter=None,
                       max_iter=None,
                       indices=None,
                       fig=None,
                       ax=None,
                       alpha=0.05,
                       transform=True):
        """

        """
        ## Get Trace
        _, data = self._get_phi_trace(epoch, k, min_iter, max_iter, transform=transform)
        if data is None:
            return None, None
        ## Compute Percentiles
        q = np.nanpercentile(data, axis=0, q=[100*alpha/2, 50, 100-(100*alpha/2)])
        ## Indices
        if indices is not None:
            q = q[:,indices]
            terms = [self.vocabulary[i] for i in indices]
        else:
            terms = self.vocabulary
        ## Summarize Bounds
        q = pd.DataFrame(data=q.T, index=terms, columns=["lower","median","upper"])
        q = q.sort_values("median", ascending=True)
        index = list(range(q.shape[0]))
        ## Create Plot
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.8))
        ax.barh(index,
                left=q["lower"],
                width=q["upper"]-q["lower"],
                color="C0",
                alpha=0.3,
                label="{:.0f}% C.I.".format(100-(alpha*100)))
        ax.scatter(q["median"],
                   index,
                   color="C0",
                   alpha=0.8,
                   s=50)
        ax.set_ylim(-.5, max(index)+.5)
        ax.set_xlim(left=max(q["lower"].min() - 0.01, 0))
        ax.set_yticks(index)
        ax.legend(loc="lower right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticklabels(q.index.tolist(), fontsize=8)
        ax.set_xlabel("Loading" if transform else "Parameter", fontweight="bold", fontsize=14)
        ax.tick_params(axis="x", labelsize=14)
        fig.tight_layout()
        return fig, ax

    def _plot_phi_trace(self,
                        epoch,
                        k,
                        indices=None,
                        fig=None,
                        ax=None,
                        transform=True):
        """

        """
        ## Get Data
        epochs, data = self._get_phi_trace(epoch, k, None, None, transform=transform)
        if data is None:
            return None, None
        ## Isolate Desired Components
        if indices is None:
            indices = range(data.shape[1])
        ## Create Figure
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.6))
        for term in indices:
            ax.plot(epochs, data[:,term],alpha=0.4)
        ax.set_xlim(0, max(epochs))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
        ax.set_ylabel("Sample", fontweight="bold", fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        return fig, ax

    def _plot_concentration_trace(self,
                                  parameter,
                                  epochs=None,
                                  fig=None,
                                  ax=None):
        """

        """
        if parameter not in ["alpha","gamma"]:
            raise ValueError("Parameter type not supported.")
        cache = getattr(self, f"_{parameter}_cache",None)
        if cache is None:
            return None, None
        mcmc_epochs = [c[0] for c in cache]
        data = np.stack([c[1] for c in cache]).T
        if epochs is None:
            epochs = range(data.shape[0])
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.8))
        for e in epochs:
            ax.plot(mcmc_epochs, data[e], linewidth=2, alpha=0.4)
        ax.set_xlim(0, max(mcmc_epochs))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
        ax.set_ylabel("Sample", fontweight="bold", fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        return fig, ax

    def plot_document_trace(self,
                            doc_id,
                            top_k=None,
                            fig=None,
                            ax=None):
        """

        """
        ## Get All Component Paths
        m = len(self._theta_cache)
        paths = [self._get_component_path(k) for k in range(self.theta.shape[1])]
        ## Get Data
        epochs = []
        data = []
        for k, p in enumerate(paths):
            p_epochs = np.zeros(m) * np.nan
            p_data = np.zeros(m) * np.nan
            for iteration, k_ind in enumerate(p):
                if k_ind is None:
                    continue
                mcmc_epoch, mcmc_data = self._theta_cache[iteration]
                mcmc_data = mcmc_data[doc_id,k_ind]
                p_epochs[iteration] = mcmc_epoch
                p_data[iteration] = mcmc_data
            epochs.append(p_epochs)
            data.append(p_data)
        ## Format
        epochs = np.stack(epochs)
        data = np.stack(data)
        ## Identify Top K
        components = range(data.shape[0])
        if top_k is not None:
            components = sorted(data.mean(axis=1).argsort()[-top_k:][::-1])
        ## Make Figure
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.6))
        for c in components:
            ax.plot(epochs[c], data[c], alpha=0.4, linewidth=3)
        ax.set_xlim(0, np.nanmax(epochs))
        ax.set_ylim(bottom=0, top=np.nanmax(data) + 0.05)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
        ax.set_ylabel("Sample", fontweight="bold", fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        return fig, ax    

    def plot_topic_trace(self,
                         topic_id,
                         epoch=None,
                         top_k_terms=None,
                         alpha=0.05,
                         transform=True):
        """

        """
        ## Initialize Figure
        fig, ax = plt.subplots(1, 2, figsize=(10,5.8), sharex=False, sharey=False)
        ## Get Terms
        indices = range(self.V)
        if top_k_terms is not None:
            indices = sorted(self.phi[epoch,topic_id].argsort()[-top_k_terms:])
        ## Plot Shaded Bar Plot of Terms
        fig, ax[0] = self._plot_phi_dist(epoch=epoch,
                                         k=topic_id,
                                         min_iter=None,
                                         max_iter=None,
                                         indices=indices,
                                         fig=fig,
                                         ax=ax[0],
                                         alpha=alpha,
                                         transform=transform)
        if fig is None:
            return None
        ## Plot Trace
        fig, ax[1] = self._plot_phi_trace(epoch=epoch,
                                          k=topic_id,
                                          indices=indices,
                                          fig=fig,
                                          ax=ax[1],
                                          transform=transform)
        if fig is None:
            return None
        fig.tight_layout()
        return fig, ax

    def plot_acceptance_trace(self,
                              fig=None,
                              ax=None):
        """

        """
        ## Parameter Paths
        components = range(self.phi.shape[1])
        paths = [(k,self._get_component_path(k)) for k in components]
        ## Get Data
        epochs = []
        data = []
        kvals = []
        all_epochs = set()
        for k, kpath in paths:
            kvals.append(k)
            kdata = np.zeros(len(kpath)) * np.nan
            kepochs = np.zeros(len(kpath)) * np.nan
            for j, kind in enumerate(kpath):
                if kind is None:
                    continue
                kdata[j] = self._acceptance_cache[j][1][kind]
                kepochs[j] = self._acceptance_cache[j][0]
                all_epochs.add(kepochs[j])
            epochs.append(kepochs)
            data.append(kdata)
        epochs = np.stack(epochs)
        data = np.stack(data)
        ## Acceptance Rate
        nn_by_epoch = (~np.isnan(data)).sum(axis=0)
        accept = np.nansum(data,axis=0)
        accept_rate = accept / nn_by_epoch
        accept_epochs = sorted(all_epochs)
        ## Plot
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.8))
        for component, e, d in zip(kvals, epochs, data):
            dplot = np.zeros_like(d) * np.nan
            dplot[d==1] = component
            ax.scatter(e, dplot, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(accept_epochs, accept_rate, color="black", linewidth=2, alpha=0.5, marker="x")
        for a in [ax, ax2]:
            a.set_xlim(-.5, np.nanmax(epochs)+.5)
            a.spines["top"].set_visible(False)
            a.tick_params(labelsize=12)
        ax.set_ylim(-.5, max(components) + 1)
        ax2.set_ylim(-.01, 1.01)
        ax2.set_ylabel("Acceptance Rate", fontweight="bold", fontsize=14)
        ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
        ax.set_ylabel("Component", fontweight="bold", fontsize=14)
        fig.tight_layout()
        return fig, ax

    def plot_eta_trace(self,
                       components=None,
                       fig=None,
                       ax=None):
        """

        """
        ## Get Paths
        if components is None:
            components = range(self.phi.shape[1])
        paths = [(k,self._get_component_path(k)) for k in components]
        ## Get Data
        epochs = []
        data = []
        kvals = []
        all_epochs = set()
        for k, kpath in paths:
            kvals.append(k)
            kdata = np.zeros(len(kpath)) * np.nan
            kepochs = np.zeros(len(kpath)) * np.nan
            for j, kind in enumerate(kpath):
                if kind is None:
                    continue
                kdata[j] = self._eta_cache[j][1][kind]
                kepochs[j] = self._eta_cache[j][0]
                all_epochs.add(kepochs[j])
            epochs.append(kepochs)
            data.append(kdata)
        epochs = np.stack(epochs)
        data = np.stack(data)
        ## Plot
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10,5.8))
        for _, e, d in zip(kvals, epochs, data):
            ax.plot(e, d, alpha=0.4, linewidth=2)
        ax.set_xlim(0, np.nanmax(epochs)+1)
        ax.set_ylim(0, np.nanmax(data)+0.01)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("MCMC Iteration", fontweight="bold", fontsize=14)
        ax.set_ylabel("Sample", fontweight="bold", fontsize=14)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        return fig, ax

    def plot_alpha_trace(self,
                         epochs=None):
        """

        """
        fig, ax = self._plot_concentration_trace("alpha", epochs=epochs)
        return fig, ax
    
    def plot_gamma_trace(self,
                         epochs=None):
        """

        """
        fig, ax = self._plot_concentration_trace("gamma", epochs=epochs)
        return fig, ax

    def plot_topic_evolution(self,
                             topic_id,
                             top_k_terms=None,
                             top_k_type=np.nanmean,
                             transform=True):
        """

        """
        ## Check Argument
        if topic_id >= self.phi.shape[1]:
            raise ValueError("Topic ID out of range")
        ## Get Data
        data = self.phi[:,topic_id,:]
        if transform:
            data = logistic_transform(data, axis=1, keepdims=True)
        terms = self.vocabulary if self.vocabulary is not None else list(range(data.shape[1]))
        if top_k_terms is not None:
            top_k_i = sorted(top_k_type(data, axis=0).argsort()[-top_k_terms:])
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
    
    def fit(self,
            X,
            timepoints,
            checkpoint_location=None,
            checkpoint_frequency=100):
        """

        """
        ## Set Random Seed
        _ = np.random.seed(self._seed)
        ## Vocabulary Check
        if self.vocabulary is None:
            self.vocabulary = list(range(X.shape[1]))
        ## Convert X to a List
        X = self._construct_item_frequency_list(X)
        ## Initialize Bookkeeping
        _ = self._initialize_bookkeeping(X, timepoints)
        ## Initialize Parameter Cache for Traces
        self._alpha_cache = []
        self._gamma_cache = []
        self._phi_cache = []
        self._theta_cache = []
        self._eta_cache = []
        self._acceptance_cache = []
        self._current_iteration = 0
        ## Inference Loop
        n_updates = 0
        for iteration in range(self.n_iter):
            if self.verbose:
                print("~"*50 + f"\nBeginning Iteration {iteration}\n" + "~" * 50)            
            ## Sample Batches
            sample_indices = list(range(len(X)))
            if self._batch_size is None:
                batches = [sample_indices]
            else:
                np.random.shuffle(sample_indices)
                batches = list(chunks(sample_indices, self._batch_size))
            ## Train in Batches
            for b, batch in enumerate(batches):
                ## Run One Gibbs Iteration
                _ = self._train(X, iteration, batch, b+1, iteration==self.n_iter-1)
                ## Caching
                if self.cache_rate is not None and (n_updates + 1) % self.cache_rate == 0:
                    if "theta" in self.cache_params:
                        self._theta_cache.append((n_updates, self._get_theta()))
                    if "phi" in self.cache_params:
                        self._phi_cache.append((n_updates, self.phi.copy()))
                    if "alpha" in self.cache_params:
                        self._alpha_cache.append((n_updates, self.alpha.copy()))
                    if "gamma" in self.cache_params:
                        self._gamma_cache.append((n_updates, self.gamma.copy()))
                    if "eta" in self.cache_params:
                        self._eta_cache.append((n_updates, self.m.sum(axis=0) / self.m.sum()))
                    if "acceptance" in self.cache_params:
                        self._acceptance_cache.append((n_updates, self._acceptance.copy()))
                ## Checkpoint
                if checkpoint_location is not None and (n_updates + 1) % checkpoint_frequency == 0:
                    _ = make_directory(checkpoint_location)
                    _ = self.save(f"{checkpoint_location}/model.joblib")
                ## Increase Update Counter (Minibatches)
                n_updates += 1
                self._current_iteration += 1
        ## Cache Final Theta
        self.theta = self._get_theta()
        return self
    
    def infer(self,
              X,
              timepoints=None):
        """

        """
        raise NotImplementedError("Inference without training has not yet been implemented.")
