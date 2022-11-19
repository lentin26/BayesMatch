import sys

import numpy as np
from scipy.special import gammaln
from numpy.random import multinomial
import matplotlib.pyplot as plt
import pandas as pd


class BayesMatchBase:

    def __init__(self, corpus_jobs, corpus_members, K, alpha, beta_m=0.01, beta_j=0.01, beta_c=0.01):
        """
        Create BayesMatch instance
        :param corpus_jobs: ndarray
            Job-specific data
        :param corpus_members: ndarray
            Member-specific data
        :param K: array_like
            Number of clusters
        :param alpha: array_like
            Dirichlet hyperparameter over cluster mixture
        :param beta_m: array_like
            Dirichlet hyperparameter over feature mixture
        """
        # params
        self.corpus_jobs = corpus_jobs
        self.corpus_members = corpus_members
        self.common_features = self.get_common_features()
        self.job_features = self.get_job_features()
        self.member_features = self.get_member_features()

        self.K = K
        self.alpha = alpha
        self.N_j = len(corpus_jobs)
        self.N_m = len(corpus_members)

        self.Vm = self.get_member_feature_count()
        self.Vj = self.get_job_feature_count()
        self.Vc = self.get_common_feature_count()

        self.V = self.n_features()

        # initialize count parameters
        self.members_cluster_assignment = [[0] * len(mem) for mem in corpus_members]
        self.jobs_cluster_assignment = [[0] * len(job) for job in corpus_jobs]

        self.pos_feature_assignment_m = [np.array([[0] * len(mem) for mem in corpus_members])]*self.K

        # feature x by feature v by cluster k
        self.cluster_count_m = np.zeros(self.K)
        self.cluster_count_j = np.zeros(self.K)

        # number member-specific, job-specific and common features in each corpus
        self.nm_tokens = self.get_n_member_tokens()
        self.nj_tokens = self.get_n_job_tokens()
        self.nc_tokens = self.get_n_common_tokens()

        self.feature_feature_cluster_count = np.zeros((self.V, self.V, self.K))
        # self.feature_feature_cluster_count_j = np.zeros((self.V, self.V, self.K))

        # set priors
        if isinstance(beta_m, float):
            self.beta_m = np.array([beta_m])*self.Vm
        if isinstance(beta_j, float):
            self.beta_j = np.array([beta_j])*self.Vj
        if isinstance(beta_c, float):
            self.beta_c = np.array([beta_c])*self.Vc

        self.log_likelihood_trace = []
        self.perplexity_trace = []

    def n_features(self):
        """
        Number of unique features spanning job and member corpuses
        :return:
        """
        nm = len(self.get_member_features())
        nj = len(self.get_job_features())
        nc = len(self.get_common_features())
        return nm + nj + nc

    def get_member_corpus_features(self):
        """
        Get distinct member features
        :return:
        """
        return list({x for l in self.corpus_members for x in l})

    def get_job_corpus_features(self):
        """
        Get distinct job features
        :return:
        """
        return list({x for l in self.corpus_jobs for x in l})

    def get_common_features(self):
        """
        Get distinct common feature indices
        :return:
        """
        members_feature_idx = self.get_member_corpus_features()
        jobs_feature_idx = self.get_job_corpus_features()
        common_features = np.intersect1d(members_feature_idx, jobs_feature_idx)
        return np.unique(common_features)

    def get_member_features(self):
        """
        Get distinct member features
        :return:
        """
        member_features = self.get_member_corpus_features()
        return list({x for x in member_features if x not in self.get_common_features()})

    def get_job_features(self):
        """
        Get distinct job features
        :return:
        """
        job_features = self.get_job_corpus_features()
        return list({x for x in job_features if x not in self.get_common_features()})

    def get_number_of_tokens(self):
        """
        Count number of tokens in dataset
        :return:
        """
        n_member_tokens = sum(len(l) for l in self.corpus_members)
        n_job_tokens = sum(len(l) for l in self.corpus_jobs)
        return n_member_tokens + n_job_tokens

    def get_n_member_tokens(self):
        """
        Get number of member_specific tokens
        :return:
        """
        # count common tokens from member corpus
        n_tokens = 0
        for mem in self.corpus_members:
            for word in mem:
                if word in self.member_features:
                    n_tokens += 1
        return n_tokens

    def get_n_job_tokens(self):
        """
        Get number of member_specific tokens
        :return:
        """
        n_tokens = 0
        for mem in self.corpus_jobs:
            for word in mem:
                if word in self.job_features:
                    n_tokens += 1
        return n_tokens

    def get_n_common_tokens(self):
        """
        Get number of common_specific tokens
        :return:
        """
        common_features = self.get_common_features()
        # count common tokens from member corpus
        n_tokens_m = 0
        for mem in self.corpus_members:
            for word in mem:
                if word in common_features:
                    n_tokens_m += 1

        # count common tokens from job corpus
        n_tokens_j = 0
        for job in self.corpus_jobs:
            for word in job:
                if word in common_features:
                    n_tokens_j += 1

        return n_tokens_m + n_tokens_j

    def get_member_feature_count(self):
        """
        Get unique count of members features
        :return:
        """
        return len(self.get_member_features())

    def get_job_feature_count(self):
        """
        Get unique count of jobs features
        :return:
        """
        return len(self.get_job_features())

    def get_common_feature_count(self):
        """
        Get unique count of jobs with common features
        :return:
        """
        return len(self.get_common_features())

    def get_cluster_assignment_m(self, member_idx, pos):
        """
        Returns current cluster assignment of feature in member at given position.
        """
        return self.members_cluster_assignment[member_idx][pos]

    def get_cluster_assignment_j(self, job_idx, pos):
        """
        Returns current cluster assignment of feature in member at given position.
        """
        return self.jobs_cluster_assignment[job_idx][pos]

    def update_member(self, cluster_idx, member_idx, feature_idx, pos, count):
        """
        Increases or decreases all member count parameters by given count value (+1 or -1).
        :return:
        """
        # update counts
        x = self.corpus_members[member_idx]
        self.cluster_count_m[cluster_idx] += count
        self.feature_feature_cluster_count[x, feature_idx, cluster_idx] += count
        self.members_cluster_assignment[member_idx][pos] = cluster_idx

    def update_job(self, cluster_idx, job_idx, feature_idx, pos, count):
        """
        Increases or decreases all job count parameters by given count value (+1 or -1).
        :return:
        """
        # update counts
        x = self.corpus_jobs[job_idx]
        self.cluster_count_j[cluster_idx] += count
        self.feature_feature_cluster_count[x, feature_idx, cluster_idx] += count
        self.jobs_cluster_assignment[job_idx][pos] = cluster_idx

    def members_full_conditional_posterior(self, feature_idx):
        """
        Returns full conditional distribution for given job or member index, cluster index and feature_idx.
        Returns multinomial vector for latent variable assignment of a member, which should sum to 1
        :return
        """
        # count of member assignments
        m = self.cluster_count_m
        member_assignments = m + self.alpha

        # total number of times members with feature x equal to v were assigned to k
        n_mf_k = self.feature_feature_cluster_count[self.member_features, feature_idx, :]
        # number of times member with feature x equal to u assigned to k
        n_m_k = self.feature_feature_cluster_count[self.member_features, feature_idx, :].sum(axis=0)
        # total number of times members with feature x equal to v were assigned to k
        n_cf_k = self.feature_feature_cluster_count[self.common_features, feature_idx, :]
        # number of times member with feature x equal to u assigned to k
        n_c_k = self.feature_feature_cluster_count[self.common_features, feature_idx, :].sum(axis=0)

        # ratio of x, v assigned to k vs x assigned to k
        member_ratio = (n_mf_k + self.beta_m) / (n_m_k + self.Vm * self.beta_m)
        common_ratio = (n_cf_k + self.beta_m) / (n_c_k + self.Vm * self.beta_m)

        # non-normalized conditional probability
        p_z_cond = member_assignments * member_ratio.prod(axis=0) * common_ratio.prod(axis=0)
        return p_z_cond / p_z_cond.sum()

    def jobs_full_conditional_posterior(self, feature_idx):
        """
        Returns full conditional distribution for given job or member index, cluster index and feature_idx.
        Returns multinomial vector for latent variable assignment of a member, which should sum to 1
        :return
        """
        # count of cluster assignments
        m = self.cluster_count_j
        assignments = m + self.alpha

        # number of times member with feature x equal to u assigned to k
        n_mf_k = self.feature_feature_cluster_count[self.job_features, feature_idx, :]
        # total number of times members with feature x equal to v were assigned to k
        n_m_k = self.feature_feature_cluster_count[self.job_features, feature_idx, :].sum(axis=0)
        # number of times member with feature x equal to u assigned to k
        n_cf_k = self.feature_feature_cluster_count[self.common_features, feature_idx, :]
        # total number of times members with feature x equal to v were assigned to k
        n_c_k = self.feature_feature_cluster_count[self.common_features, feature_idx, :].sum(axis=0)

        # un-normalized conditional prob
        job_ratio = (n_mf_k + self.beta_j) / (n_m_k + self.Vj * self.beta_j)
        common_ratio = (n_cf_k + self.beta_j) / (n_c_k + self.Vj * self.beta_j)

        # product of ratios
        p_z_cond = assignments * job_ratio.prod(axis=0) * common_ratio.prod(axis=0)
        return p_z_cond / p_z_cond.sum()

    def get_log_likelihood(self):
        """
        Returns joint log likelihood
        p(fm, fj, zm, zj) = p(fm, zm)p(fj, zj) = p(fm|zm)p(fj|zj)p(zm)p(zj).
        Griffiths and Steyvers (2004): Finding scientific topics
        :return:
        """
        log_likelihood = 0.0
        for z in range(self.K):  # log p(fm|z)
            # members
            log_likelihood += 2*gammaln(self.alpha.sum())
            log_likelihood -= 2*gammaln(self.alpha).sum()
            log_likelihood += gammaln(self.cluster_count_m + self.alpha).sum()
            log_likelihood -= gammaln((self.cluster_count_m + self.alpha).sum())
            log_likelihood += gammaln(self.cluster_count_j + self.alpha).sum()
            log_likelihood -= gammaln((self.cluster_count_j + self.alpha).sum())
        # for v in range(self.Vm):
        #     log_likelihood += gammaln(self.beta_m.sum())
        #     log_likelihood -= gammaln(self.beta_m).sum()
        #     log_likelihood += gammaln(self.beta_c.sum())
        #     log_likelihood -= gammaln(self.beta_c).sum()
        # for v in range(self.Vc):
        #     log_likelihood += gammaln(self.beta_j.sum())
        #     log_likelihood -= gammaln(self.beta_j).sum()
        #     log_likelihood += gammaln(self.beta_c.sum())
        #     log_likelihood -= gammaln(self.beta_c).sum()
        for z in range(self.K):
            for x in self.get_member_features():
                log_likelihood += gammaln(self.feature_feature_cluster_count[x, :, z] + self.beta_m).sum()
                log_likelihood -= gammaln((self.feature_feature_cluster_count[x, :, z] + self.beta_m).sum())
            for x in self.get_job_features():
                log_likelihood += gammaln(self.feature_feature_cluster_count[x, :, z] + self.beta_m).sum()
                log_likelihood -= gammaln((self.feature_feature_cluster_count[x, :, z] + self.beta_m).sum())
        return log_likelihood

    def trace_metrics(self):
        """
        Traces metrics to ensure convergency.
        """
        log_likelihood = self.get_log_likelihood()
        n = self.get_number_of_tokens()
        perplexity = np.exp(-log_likelihood / n)
        self.log_likelihood_trace.append(log_likelihood)
        self.perplexity_trace.append(perplexity)

    def plot_metrics(self):
        """
        Plots log likelihood and perplexity trace.
        """
        data = zip(self.log_likelihood_trace, self.perplexity_trace)
        df = pd.DataFrame(data, columns=["Log Likelihood", "Perplexity"])
        df.plot(legend=True, title="Convergence Metrics", figsize=(15, 5), subplots=True, layout=(1, 2))
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # generate random dataset
    n = 50
    cj = np.random.multinomial(100, pvals=[1 / n] * n).reshape(5, 10)
    cm = np.random.multinomial(n=100, pvals=[1 / n] * n).reshape(5, 10) + 5

    K = 100
    base = BayesMatchBase(corpus_jobs=cj, corpus_members=cm, K=K, alpha=[1])







