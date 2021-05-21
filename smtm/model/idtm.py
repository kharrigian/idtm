
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
from hmmlearn.hmm import GaussianHMM

## Local
from .base import TopicModel
from .helpers import (sample_categorical,
                      sample_multinomial,
                      logistic_transform)

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
                 q_dim=1,
                 alpha_filter=1,
                 gamma_filter=1,
                 n_filter=10,
                 threshold=0.01,
                 k_filter_frequency=None,
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
        self._alpha_0 = (alpha_0_a, alpha_0_b)
        self._gamma_0 = (gamma_0_a, gamma_0_b)
        self._rho_0 = rho_0
        self._sigma_0 = sigma_0
        self._alpha_filter = alpha_filter
        self._gamma_filter = gamma_filter
        self._n_filter = n_filter
        self._q_dim = q_dim
        self._threshold = threshold
        self._k_filter_freq = k_filter_frequency
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
    
    def _normalize(self,
                   a,
                   **kwargs):
        """

        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        asum = a.sum(**kwargs)
        a_norm = np.divide(a,
                           asum,
                           out=np.zeros_like(a),
                           where=asum>0)
        return a_norm

    def _initialize_bookkeeping(self,
                                X,
                                timepoints):
        """

        """
        ## Vocabulary Size
        self.V = len(self.vocabulary)
        ## Initialize Random Table Assignments For Each Word [n x w_n]
        self.word2table = [[np.random.randint(self._initial_m) for _ in x] for x in X]
        ## Initialize Random Dish Assignments for Each Table [n x m_n]
        self.table2dish = [[np.random.randint(self._initial_k) for _ in range(self._initial_m)] for _ in X]
        ## Epoch Associated with Each Document
        self.rest2epoch = [[] for t in range(self._t)]
        for i, t in enumerate(timepoints):
            self.rest2epoch[t].append(i)
        ## Space for Sampling
        initial_kmax = max(self.V // 2, self._initial_k)
        initial_mmax = max(max(list(map(len,X))) // 2, self._initial_m)
        ## Table and Component Counts (m: [t x k_max], n: [n x m_max])
        self.m = np.array([[0 for _ in range(initial_kmax)] for _ in range(self._t)], dtype=np.int)
        self.m_kt_prime = np.zeros(self.m.shape)
        self.n = np.array([[0 for _ in range(initial_mmax)] for _ in X], dtype=np.int)
        for epoch, documents in enumerate(self.rest2epoch):
            for d in documents:
                for dish_k in self.table2dish[d]:
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
        self._H = stats.multivariate_normal([0] * self.V, self._sigma_0)
        ## Initialize Components [t x k_max x V]
        self.phi = np.zeros((self._t, initial_kmax, self.V))
        self.phi[0] = self._H.rvs(initial_kmax)
        if initial_kmax == 1:
            self.phi[0] = self.phi[0].reshape(1,-1)
        for epoch in range(1, self._t):
            self.phi[epoch] = np.zeros(self.phi[0].shape)
            for k, comp in enumerate(self.phi[epoch-1]):
                self.phi[epoch][k] = stats.multivariate_normal(comp, self._rho_0).rvs()
        self.phi = np.stack(self.phi)
        ## Transform Components
        self.phi_T = logistic_transform(self.phi, axis=2, keepdims=True)
        ## Initialize Word Counts 
        ## Z: [t x k_max x V] frequency of each word for each k for each t
        ## v: [n x m_max x V]: frequency of each word in each table for each document
        self.Z = np.zeros_like(self.phi, dtype=np.int)
        self.v = np.zeros((len(X), initial_mmax, self.V), dtype=np.int)
        for epoch, documents in enumerate(self.rest2epoch):
            for d in documents:
                for w_d, t_d in zip(X[d], self.word2table[d]):
                    k_d = self.table2dish[d][t_d]
                    self.Z[epoch,k_d,w_d] += 1
                    self.v[d, t_d, w_d] += 1
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
        q_ll = 0 if phi_k.shape[0] == 1 else q.score(phi_k)
        q_star_ll = 0 if phi_k.shape[0] == 1 else q.score(phi_k_star)
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
            emission_star_ll.append(stats.multinomial(n_kt, phi_k_star_smooth[t]).logpmf(v_kt))
            transition_star_ll.append(stats.multivariate_normal(phi_k_star[t-1],self._rho_0).logpdf(phi_k_star[t]))
            ## Existing Data Likelihood
            emission_star_ll.append(stats.multinomial(n_kt, phi_k_smooth[t]).logpmf(v_kt))
            transition_ll.append(stats.multivariate_normal(phi_k[t-1],self._rho_0).logpdf(phi_k[t]))
        ## Compute Ratio
        if v_k.shape[0] > 1:
            numerator = h_star_ll + logsumexp(emission_star_ll + transition_star_ll) + q_ll
            denominator = h_ll + logsumexp(emission_ll + transition_ll) + q_star_ll
        else:
            numerator = h_star_ll + q_ll
            denominator = h_ll + q_star_ll
        ratio_ll = numerator - denominator
        ratio = min(1, np.exp(ratio_ll))
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

    def _train(self,
               X,
               iteration,
               is_last=False):
        """

        """
        ## Check for Component Filtering
        k_filter_on = False
        if (self._k_filter_freq is not None and iteration % self._k_filter_freq == 0) or is_last:
            mdist = self.m.sum(axis=0) / self.m.sum()
            k_filter = (mdist >= self._threshold).astype(int)
            k_filter_on = True
        ## Track Where Components Are Created
        k_created = [0,0]
        ####### Step 1: Sample Topic k_tdb for table tdb
        ## Reset m'
        self.m_kt_prime = np.zeros(self.m.shape)
        ## Iterate over time periods
        for epoch, epoch_docs in enumerate(self.rest2epoch):
            ## Indices of Documents in Epoch
            epoch_docs = self.rest2epoch[epoch]
            ## Compute m_kt' for the Epoch
            if epoch == 0:
                pass
            elif (epoch + 1) < self._delta:
                self.m_kt_prime[epoch] = (self.m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0)
            elif (epoch + 1) >= self._delta:
                self.m_kt_prime[epoch] = (self.m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0) - \
                                        self.m[epoch-(self._delta+1)] * np.exp(-(self._delta + 1) / self._lambda_0)
            ## Cycle Through Documents
            for document in epoch_docs:
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
                    f_v_tdb_used = np.array([stats.multinomial(n_tdb, p).pmf(v_tdk) for p in self.phi_T[epoch]])
                    if epoch > 0:
                        f_v_tdb_exists = np.array([stats.multinomial(n_tdb, p).pmf(v_tdk) for p in self.phi_T[epoch-1]])
                    else:
                        f_v_tdb_exists = np.zeros_like(f_v_tdb_used)
                    f_v_tdb_new = np.array([stats.multinomial(n_tdb, p).pmf(v_tdk) for p in self._phi_aux_T])
                    ## Probabilities
                    p_k_used = (self.m[epoch] + self.m_kt_prime[epoch]) * f_v_tdb_used
                    p_k_exist = self.m_kt_prime[epoch] * f_v_tdb_exists
                    p_k_not_new = np.where([i in k_used for i in range(p_k_used.shape[0])],
                                            p_k_used,
                                            p_k_exist)
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
                        ## Check To See if Expansion Need
                        if k_new_ind >= self.m.shape[1]:
                            exp_size = self.m.shape[1] // 2
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
        for epoch, epoch_docs in enumerate(self.rest2epoch):
            for document in epoch_docs:
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
                            exp_size = self.n.shape[1] // 4
                            self.n = np.hstack([self.n, np.zeros((self.n.shape[0],exp_size), dtype=np.int)])
                            self.v = np.hstack([self.v, np.zeros((self.v.shape[0],exp_size,self.v.shape[2]),dtype=np.int)])
                        ## Probability of Word For a New Table under Each Component
                        p_k_used = (self.m_kt_prime[epoch] + self.m[epoch]) * self.phi_T[epoch,:,x_tdi]
                        if epoch > 0:
                            p_k_exists = self.m_kt_prime[epoch] * self.phi_T[epoch-1,:,x_tdi]
                        else:
                            p_k_exists = np.zeros_like(p_k_used)
                        p_k_not_new = np.where([i in k_used for i in range(p_k_used.shape[0])],
                                                p_k_used,
                                                p_k_exists)
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
                            ## Check Cache Size (Components)
                            if k_new_ind >= self.m.shape[1]:
                                exp_size = self.m.shape[1] // 2
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
        _ = self._clean_up()
        ####### Step 5: Sample Components phi_tk using Z
        n_accept = [0,0]
        for k in range(self.phi.shape[1]):
            ## Isolate Existing Variables
            phi_k = self.phi[:,k,:]
            v_k = self.Z[:,k,:]
            ## Filter by Lifespan
            born, die = self.K_life[k]
            phi_k = phi_k[born:die+1]
            v_k = v_k[born:die+1]
            ## Generate Proposal Distribution and Sample from It
            if phi_k.shape[0] == 1:
                q = None
                phi_k_star = stats.multivariate_normal(phi_k[0], self._sigma_0).rvs()
            else:
                nn_phi = len(phi_k.sum(axis=1).nonzero()[0])
                q = GaussianHMM(n_components=max(min(self._q_dim, nn_phi // 3), 1),
                                covariance_type="diag",
                                n_iter=10,
                                means_prior=0,
                                covars_prior=self._sigma_0,
                                random_state=self._seed,
                                verbose=False)
                q = q.fit(phi_k)
                phi_k_star, _ = q.sample(phi_k.shape[0])
            ## MH Step
            phi_k, accept = self._metropolis_hastings(phi_k,
                                                      phi_k_star,
                                                      v_k,
                                                      q)
            n_accept[0] += accept
            n_accept[1] += 1
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
            print("\nIteration {} Complete\n".format(iteration+1)+"~"*50)
            print("Acceptance:", n_accept[0] / n_accept[1])
            print("# Components:", max(list(map(max,self.table2dish))))
            print("Max # Tables:", max(list(map(max,self.word2table))))
            print("Number of New Components Per Sampling Stage:", k_created)
            print("Alpha:", self.alpha)
            print("Gamma", self.gamma)
            print("Eta:", self.m.sum(axis=0))

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
            ## Run One Gibbs Iteration
            _ = self._train(X, iteration, iteration==self.n_iter-1)
        return self


