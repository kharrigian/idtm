
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
        self.word2table = [[np.random.randint(self._initial_k) for _ in x] for x in X]
        ## Initialize Random Dish Assignments for Each Table [n x m_n]
        self.table2dish = [[np.random.randint(self._initial_m) for _ in range(self._initial_k)] for _ in X]
        ## Epoch Associated with Each Document
        self.rest2epoch = [[] for t in range(self._t)]
        for i, t in enumerate(timepoints):
            self.rest2epoch[t].append(i)
        ## Component Counts For Each Epoch [t x 1]
        self.K_t = np.zeros(self._t, dtype=np.int)
        k_present = []
        self.K_born = []
        for epoch, documents in enumerate(self.rest2epoch):
            epoch_comp_counts = Counter()
            for d in documents:
                for comp_k in self.table2dish[d]:
                    epoch_comp_counts[comp_k] += 1
            self.K_t[epoch] = len(epoch_comp_counts)
            k_present.append(set(epoch_comp_counts.keys()))
            if epoch == 0:
                self.K_born.append(k_present[-1])
            else:
                new_present = set()
                for p in k_present[:-1]:
                    new_present.update(k_present[-1] - p)
                self.K_born.append(new_present)
        ## Space for Sampling
        initial_kmax = max(self.V // 2, max(self.K_t))
        initial_mmax = max(max(list(map(len,X))) // 2, self._initial_m)
        ## Table and Component Counts (m: [t x k_max], n: [n x m_max])
        self.m = np.array([[0 for _ in range(initial_kmax)] for _ in range(self._t)], dtype=np.int)
        self.n = np.array([[0 for _ in range(initial_mmax)] for _ in X], dtype=np.int)
        for epoch, documents in enumerate(self.rest2epoch):
            for d in documents:
                for dish_k in self.table2dish[d]:
                    self.m[epoch][dish_k] += 1
                for table_k in self.word2table[d]:
                    self.n[d][table_k] += 1
        ## Initialize Components [t x k_max x V]
        self.phi = [None for _ in range(self._t)]
        self.phi[0] = stats.dirichlet([self._sigma_0] * self.V).rvs(initial_kmax)
        for epoch in range(1, self._t):
            self.phi[epoch] = np.zeros(self.phi[0].shape)
            for k, comp in enumerate(self.phi[epoch-1]):
                self.phi[epoch][k] = stats.dirichlet(comp + self._rho_0).rvs()
        self.phi = np.stack(self.phi)
        ## Active Components in Each Epoch [t x k_t]
        self._live_k = [list(range(k)) for k in self.K_t]
        ## Initialize Word Counts (Z: [t x k_max x V] frequency of each word for each k for each t)
        ## v: [n x m_max x V]: frequency of each word in each table for each document
        self.Z = np.zeros_like(self.phi, dtype=np.int)
        self.v = np.zeros((len(X), initial_mmax, self.V), dtype=np.int)
        for epoch, documents in enumerate(self.rest2epoch):
            for d in documents:
                for w_d, t_d in zip(X[d], self.word2table[d]):
                    k_d = self.table2dish[d][t_d]
                    self.Z[epoch,k_d,w_d] += 1
                    self.v[d, t_d, w_d] += 1
        ## Concetration Parameters
        self.gamma = stats.gamma(self._gamma_0[0],scale=1/self._gamma_0[1]).rvs()
        self.alpha = stats.gamma(self._alpha_0[0],scale=1/self._alpha_0[1]).rvs()
        ## Initialize Auxiliary Variable Sampler
        self._phi_aux_sampler = stats.dirichlet([self._sigma_0] * self.V)
        self._phi_aux = self._phi_aux_sampler.rvs(self._q)

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

    def _metropolis_hastings(self,
                             phi_k,
                             phi_k_star,
                             v_k,
                             q,
                             smoothing=1e-100):
        """
        Compute Acceptance Ratio for Linear Emission Model and return
        selected sample

        Args:
            phi_k (2d-array): Current parameters
            phi_k_star (2d-array): Proposed parameters
            v_k (2d-array): Current counts
            q (stats frozen distribution): proposed distribution, list if evolves over epoch

        Returns:
            phi_k_selected (2d-array), phi_k_T_selected (2d-array), accept (bool)
        """
        ## Static vs. Epoch Proposal
        if not isinstance(q, list):
            q = [q for _ in range(phi_k.shape[0])]
        ## Smooth to Avoid Underflow Errors
        phi_k_smooth = (phi_k + smoothing) / (phi_k +smoothing).sum(axis=1,keepdims=True)
        phi_k_star_smooth = (phi_k_star + smoothing) / (phi_k_star + smoothing).sum(axis=1,keepdims=True)
        ## Compute Initial State Probability
        H = stats.dirichlet([self._sigma_0] * self.V)
        h_star_ll = H.logpdf(phi_k_star_smooth[0])
        h_ll = H.logpdf(phi_k_smooth[0])
        ## Compute Transition Probabilities
        transition_ll = 0
        transition_star_ll = 0
        proposal_ll = 0
        proposal_star_ll = 0
        for t, v_kt in enumerate(v_k):
            ## Proposal Data Likelihood
            transition_star_ll += (v_kt * self._log(phi_k_star_smooth[t])).sum()
            transition_star_ll += 0 if t == 0 else stats.dirichlet(phi_k_star_smooth[t-1]).logpdf(phi_k_star_smooth[t])
            ## Existing Data Likelihood
            transition_ll += (v_kt * self._log(phi_k_smooth[t])).sum()
            transition_ll += 0 if t == 0 else stats.dirichlet(phi_k_smooth[t-1]).logpdf(phi_k_smooth[t])
            ## Proposal Component Density
            proposal_ll += q[t].logpdf(phi_k_smooth[t])
            proposal_star_ll += q[t].logpdf(phi_k_star_smooth[t])
        ## Compute Ratio
        ratio_ll = h_star_ll + transition_star_ll + proposal_ll - \
                    (h_ll + transition_ll + proposal_star_ll)
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
        self.phi = self.phi[:,nonempty_components,:]
        self.Z = self.Z[:,nonempty_components,:]
        ## Cycle Through Epochs
        for epoch, documents in enumerate(self.rest2epoch):
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
            ####### Step 1: Sample Topic k_tdb for table tdb
            print(1)
            ## Initialize m'
            m_kt_prime = np.zeros(self.m.shape)
            ## Iterate over time periods
            for epoch, epoch_docs in enumerate(self.rest2epoch):
                ## Indices of Documents in Epoch
                epoch_docs = self.rest2epoch[epoch]
                ## Compute m_kt' for the Epoch
                if epoch == 0:
                    pass
                elif (epoch + 1) < self._delta:
                    m_kt_prime[epoch] = (m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0)
                elif (epoch + 1) >= self._delta:
                    m_kt_prime[epoch] = (m_kt_prime[epoch-1] + self.m[epoch-1]) * np.exp(-1 / self._lambda_0) - \
                                         self.m[epoch-(self._delta+1)] * np.exp(-(self._delta + 1) / self._lambda_0)
                ## Cycle Through Documents
                for document in epoch_docs:
                    for table, dish in enumerate(self.table2dish[document]):
                        ## Get Words Indices at Table
                        w_tdb = [w for w, t in zip(X[document], self.word2table[document]) if t == table]
                        ## Remove Dish from Counts for the Epoch
                        self.m[epoch, dish] -= 1
                        self.Z[epoch, dish] -= self.v[document][table]
                        ## Identify Topics (Used in Epoch or Existing Previously)
                        k_used = set(self.m[epoch].nonzero()[0])
                        k_exists = set(m_kt_prime[epoch].nonzero()[0]) - k_used
                        max_k = max(k_used | k_exists)
                        ## Get Conditional Probability of Data Given Table Component
                        f_v_tdb_used = np.exp(self._log(self.phi[epoch,:,w_tdb]).sum(axis=0))
                        if epoch > 0:
                            f_v_tdb_exists = np.exp(self._log(self.phi[epoch-1,:,w_tdb]).sum(axis=0))
                        else:
                            f_v_tdb_exists = np.zeros_like(f_v_tdb_used)
                        f_v_tdb_new = np.exp(self._log(self._phi_aux[:,w_tdb]).sum(axis=1))
                        ## Probabilities
                        p_k_used = (self.m[epoch] + m_kt_prime[epoch]) * f_v_tdb_used
                        p_k_exist = m_kt_prime[epoch] * f_v_tdb_exists
                        p_k_not_new = np.where([i in k_used for i in range(p_k_used.shape[0])],
                                                p_k_used,
                                                p_k_exist)
                        p_k_new = self.gamma / self._q * f_v_tdb_new
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
                                self.phi[epoch,k_sample] = self.phi[epoch-1,k_sample]
                        ## Case 2: Sampled Topic is Completely New
                        else:
                            ## Figure Out Which Auxiliary Component Was Selected
                            k_aux = k_sample - p_k_not_new.shape[0]
                            k_new_ind = max_k + 1
                            ## Check To See if Expansion Need
                            if k_new_ind >= self.m.shape[1]:
                                exp_size = self.m.shape[1] // 2
                                self.m = np.hstack([self.m, np.zeros((self.m.shape[0], exp_size), dtype=np.int)])
                                m_kt_prime = np.hstack([m_kt_prime, np.zeros((m_kt_prime.shape[0], exp_size), dtype=np.int)])
                                self.Z = np.hstack([self.Z, np.zeros((self.Z.shape[0], exp_size, self.Z.shape[2]), dtype=np.int)])
                                self.phi = np.hstack([self.phi, np.zeros((self.phi.shape[0], exp_size, self.phi.shape[2]), dtype=np.int)])
                            ## Update Counts
                            self.m[epoch, k_new_ind] += 1
                            self.table2dish[document][table] = k_new_ind
                            self.Z[epoch, k_new_ind] += self.v[document][table]
                            ## Update Component
                            self.phi[epoch,k_new_ind] = self._phi_aux[k_aux]
                            ## Sample New Auxiliary Component
                            self._phi_aux[k_aux] = self._phi_aux_sampler.rvs()
                            ## Add to Live Components
                            self._live_k[epoch].append(k_new_ind)
                            self.K_t[epoch] += 1
            ####### Step 2: Sample a Table b_tdi for Each Word x_tdi
            print(2)
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
                        f_x_tdi = self.phi[epoch, self.table2dish[document], x_tdi]
                        p_x_tdi = self.n[document,:f_x_tdi.shape[0]] * f_x_tdi
                        ## Either Sample An Existing Table or a New Table
                        p_b_0 = np.hstack([p_x_tdi, np.r_[self.alpha]])
                        p_b_0 = p_b_0 / p_b_0.sum()
                        b_0 = sample_categorical(p_b_0)
                        ## Case 1: Existing Table
                        if b_0 < p_b_0.shape[0] - 1:
                            ## Table Component
                            k_sampled = self.table2dish[document][b_0]
                            ## Update Counts
                            self.n[document, b_0] += 1
                            self.Z[epoch,k_sampled,x_tdi] += 1
                            self.word2table[document][i] = b_0
                            self.v[document,b_0,x_tdi] += 1
                        ## Case 2: New Table
                        else:
                            ## Identify Topics (Used in Epoch or Existing Previously)
                            k_used = set(self.m[epoch].nonzero()[0])
                            k_exists = set(m_kt_prime[epoch].nonzero()[0]) - k_used
                            max_k = max(k_used | k_exists)
                            ## Check Cache Size (Tables)
                            if b_0 >= self.n.shape[1]:
                                exp_size = self.n.shape[1] // 4
                                self.n = np.hstack([self.n, np.zeros((self.n.shape[0],exp_size), dtype=np.int)])
                                self.v = np.hstack([self.v, np.zeros((self.v.shape[0],exp_size,self.v.shape[2]),dtype=np.int)])
                            ## Probability of Word For a New Table under Each Component
                            p_k_used = (m_kt_prime[epoch] + self.m[epoch]) * self.phi[epoch,:,x_tdi]
                            if epoch > 0:
                                p_k_exists = m_kt_prime[epoch] * self.phi[epoch-1,:,x_tdi]
                            else:
                                p_k_exists = np.zeros_like(p_k_used)
                            p_k_not_new = np.where([i in k_used for i in range(p_k_used.shape[0])],
                                                   p_k_used,
                                                   p_k_exists)
                            p_k_new = self.gamma / self._q * self._phi_aux[:,x_tdi]
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
                                ## Update Component for Unused Component
                                if k_sampled not in k_used:
                                    self.phi[epoch,k_sampled] = self.phi[epoch-1,k_sampled]
                            ## Case 2: Sampled a New Component
                            else:
                                ## Find Appropriate Auxiliary Variable
                                k_aux = k_sampled - p_k_not_new.shape[0]
                                k_new_ind = max_k + 1
                                ## Check Cache Size (Components)
                                if k_new_ind >= self.m.shape[1]:
                                    exp_size = self.m.shape[1] // 2
                                    self.m = np.hstack([self.m, np.zeros((self.m.shape[0], exp_size), dtype=np.int)])
                                    m_kt_prime = np.hstack([m_kt_prime, np.zeros((m_kt_prime.shape[0], exp_size), dtype=np.int)])
                                    self.Z = np.hstack([self.Z, np.zeros((self.Z.shape[0], exp_size, self.Z.shape[2]), dtype=np.int)])
                                    self.phi = np.hstack([self.phi, np.zeros((self.phi.shape[0], exp_size, self.phi.shape[2]), dtype=np.int)])
                                ## Update counts
                                self.m[epoch, k_new_ind] += 1
                                self.Z[epoch, k_new_ind, x_tdi] += 1
                                self.v[document,b_0,x_tdi] += 1
                                self.n[document,b_0] += 1
                                self.table2dish[document].append(k_new_ind)
                                self.word2table[document][i] = b_0
                                ## Add To Live Components
                                self.K_t[epoch] += 1
                                self._live_k[epoch].append(k_new_ind)
                                ## Update Component
                                self.phi[epoch,k_new_ind] = self._phi_aux[k_aux]
                                ## Sample New Auxiliary Component
                                self._phi_aux[k_aux] = self._phi_aux_sampler.rvs()
            ## Some Data Information
            n_J = self.n.sum(axis=1) ## Number of Words Per Document
            m_J = (self.n != 0).sum(axis=1) ## Number of Tables per Document
            T = self.m.sum() ## Total Number of Tables
            K = self.K_t.max() ## Maximum Number of Components         
            ####### Step 3: Sample Concetration Parameter Alpha
            print(3)
            w_j = stats.beta(self.alpha + 1, n_J).rvs()
            p_s_j = (n_J * (self._alpha_0[0] - np.log(w_j))) / (self._alpha_0[0] + m_J - 1 + n_J * (self._alpha_0[1] - np.log(w_j)))
            s_j = stats.bernoulli(p_s_j).rvs()
            self.alpha = stats.gamma(self._alpha_0[0] + self.m.sum() - s_j.sum(), scale = 1 / (self._alpha_0[1] - np.log(w_j).sum())).rvs()
            ####### Step 4: Sample Concentration Parameter Gamma
            print(4)
            eta = stats.beta(self.gamma+1, T).rvs()
            i_gamma_mix = int(np.random.uniform(0,1) > (self._gamma_0[0] + K - 1)/(self._gamma_0[0] + K - 1 + T*(self._gamma_0[1] - np.log(eta))))
            if i_gamma_mix == 0:
                self.gamma = stats.gamma(self._gamma_0[0] + K, scale = 1 / (self._gamma_0[1] - np.log(eta))).rvs()
            elif i_gamma_mix == 1:
                self.gamma = stats.gamma(self._gamma_0[0] + K - 1, scale = 1 / (self._gamma_0[1] - np.log(eta))).rvs()


            # ####### Step 5: Sample Components phi_tk using Z
            # print(5)
            # n_accept = [0,0]
            # for k in range(self.phi.shape[1]):
            #     ## Isolate Existing Variables
            #     phi_k = self.phi[:,k,:]
            #     v_k = self.Z[:,k,:]
            #     ## Generate Proposal Sample
            #     q = [stats.dirichlet(p + v + self._rho_0) for p, v in zip(phi_k, v_k)]
            #     phi_k_star = np.vstack([_q.rvs() for _q in q])
            #     ## MH Step
            #     phi_k, accept = self._metropolis_hastings(phi_k,
            #                                               phi_k_star,
            #                                               v_k,
            #                                               q)
            #     n_accept[0] += accept
            #     n_accept[1] += 1
            #     ## Update Parameters Based on MH Outcome
            #     self.phi[:,k,:] = phi_k
            # print(n_accept[0] / n_accept[1])
            # print(self.phi.shape)
            # print(self.m.shape)
            # print(self.n.shape)
            ## Clean Up (Remove Empty Components and Tables)
            _ = self._clean_up()

        return self


