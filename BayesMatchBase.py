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
        self.job_features = self.common_features
        self.member_features = self.get_member_features()
        # self.job_features = [x for x in self.get_job_features() if x not in self.common_features]
        # self.member_features = [x for x in self.get_member_features() if x not in self.common_features]

        self.K = K
        self.alpha = alpha
        self.N_j = len(corpus_jobs)
        self.N_m = len(corpus_members)

        self.n_m_f = self.get_member_feature_count()
        self.n_j_f = self.get_job_feature_count()
        self.n_c_f = self.get_common_feature_count()

        # initialize count parameters
        self.members_feature_cluster_assignment = [[0] * len(mem) for mem in corpus_members]
        self.jobs_feature_cluster_assignment = [[0] * len(job) for job in corpus_jobs]

        # feature x by feature v by cluster k
        self.Vm = self.n_m_f + self.n_c_f
        self.Vj = self.n_j_f + self.n_c_f
        self.cluster_count_m = np.zeros(self.K)
        self.cluster_count_j = np.zeros(self.K)
        self.feature_feature_cluster_count_m = np.zeros((self.Vm, self.Vm, self.K,))
        self.feature_feature_cluster_count_j = np.zeros((self.Vj, self.Vj, self.K))

        # set priors
        if isinstance(beta_m, float):
            self.beta_m = np.array([beta_m])*self.n_m_f
        if isinstance(beta_j, float):
            self.beta_j = np.array([beta_j])*self.n_j_f
        if isinstance(beta_c, float):
            self.beta_c = np.array([beta_c])*self.n_c_f

        self.log_likelihood_trace = []
        self.perplexity_trace = []

    def get_common_features(self):
        """
        Get distinct common feature indices
        :return:
        """
        members_feature_idx = self.get_member_features()
        jobs_feature_idx = self.get_job_features()
        return np.intersect1d(members_feature_idx, jobs_feature_idx)

    def get_number_of_tokens(self):
        """
        Count number of tokens in dataset
        :return:
        """
        n_member_tokens = sum(len(l) for l in self.corpus_members)
        n_job_tokens = sum(len(l) for l in self.corpus_jobs)
        return n_member_tokens + n_job_tokens

    def get_member_features(self):
        """
        Get distinct member features
        :return:
        """
        return list({x for l in self.corpus_members for x in l})

    def get_job_features(self):
        """
        Get distinct job features
        :return:
        """
        return list({x for l in self.corpus_jobs for x in l})

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
        return self.members_feature_cluster_assignment[member_idx][pos]

    def get_cluster_assignment_j(self, job_idx, pos):
        """
        Returns current cluster assignment of feature in member at given position.
        """
        return self.jobs_feature_cluster_assignment[job_idx][pos]

    def update_member(self, cluster_idx, member_idx, feature_idx, pos, count):
        """
        Increases or decreases all member count parameters by given count value (+1 or -1).
        :return:
        """
        self.cluster_count_m[cluster_idx] += count
        member_feature_idx = self.member_features
        self.feature_feature_cluster_count_m[member_feature_idx, feature_idx, cluster_idx] += count
        self.members_feature_cluster_assignment[member_idx][pos] = cluster_idx

    def update_job(self, cluster_idx, job_idx, feature_idx, pos, count):
        """
        Increases or decreases all job count parameters by given count value (+1 or -1).
        :return:
        """
        self.cluster_count_j[cluster_idx] += count
        member_feature_idx = self.job_features
        self.feature_feature_cluster_count_j[member_feature_idx, feature_idx, cluster_idx] += count
        self.jobs_feature_cluster_assignment[job_idx][pos] = cluster_idx

    def members_full_conditional_posterior(self, member_idx, feature_idx):
        """
        Returns full conditional distribution for given job or member index, cluster index and feature_idx.
        Returns multinomial vector for latent variable assignment of a member, which should sum to 1
        :return
        """
        # count of data points currently assigned to cluster k
        common_features = self.common_features
        member_features = self.member_features

        m = self.cluster_count_m

        # number of times member with feature x equal to u assigned to k
        n_mf_k = self.feature_feature_cluster_count_m[member_features, feature_idx, :]
        # total number of times members with feature x equal to v were assigned to k
        n_m_k = self.feature_feature_cluster_count_m[member_features, feature_idx, :].sum(axis=0)
        # number of times member with feature x equal to u assigned to k
        n_cf_k = self.feature_feature_cluster_count_m[common_features, feature_idx, :]
        # total number of times members with feature x equal to v were assigned to k
        n_c_k = self.feature_feature_cluster_count_m[common_features, feature_idx, :].sum(axis=0)

        # un-normalized conditional prob
        member_ratio = (n_mf_k + self.beta_m) / (n_m_k + self.Vm * self.beta_m)
        common_ratio = (n_cf_k + self.beta_m) / (n_c_k + self.Vm * self.beta_m)
        member_assignments = m + self.alpha

        p_z_cond = member_assignments * member_ratio.prod(axis=0) * common_ratio.prod(axis=0)
        return p_z_cond / p_z_cond.sum()

    def jobs_full_conditional_posterior(self, feature_idx):
        """
        Returns full conditional distribution for given job or member index, cluster index and feature_idx.
        Returns multinomial vector for latent variable assignment of a member, which should sum to 1
        :return
        """
        # count of data points currently assigned to cluster k
        common_features = self.common_features
        job_features = self.job_features
        m = self.cluster_count_j

        # number of times member with feature x equal to u assigned to k
        n_mf_k = self.feature_feature_cluster_count_m[job_features, feature_idx, :]
        # total number of times members with feature x equal to v were assigned to k
        n_m_k = self.feature_feature_cluster_count_m[job_features, feature_idx, :].sum(axis=0)
        # number of times member with feature x equal to u assigned to k
        n_cf_k = self.feature_feature_cluster_count_m[common_features, feature_idx, :]
        # total number of times members with feature x equal to v were assigned to k
        n_c_k = self.feature_feature_cluster_count_m[common_features, feature_idx, :].sum(axis=0)

        # compute ratios
        member_ratio = (n_mf_k + self.beta_m) / (n_m_k + self.Vj*self.beta_m)
        common_ratio = (n_cf_k + self.beta_m) / (n_c_k + self.Vj*self.beta_m)
        member_assignments = m + self.alpha

        # un-normalized conditional prob
        p_z_cond = member_assignments * member_ratio.prod(axis=0) * common_ratio.prod(axis=0)
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
            log_likelihood += gammaln(self.alpha.sum())
            log_likelihood -= gammaln(self.alpha).sum()
            log_likelihood += gammaln(self.cluster_count_m + self.alpha).sum()
            log_likelihood -= gammaln((self.cluster_count_m + self.alpha).sum())
        for v in range(self.Vm):
            log_likelihood += gammaln(self.beta_m.sum())
            log_likelihood -= gammaln(self.beta_m).sum()
            log_likelihood += gammaln(self.beta_c.sum())
            log_likelihood -= gammaln(self.beta_c).sum()
        for z in range(self.K):
            for x in range(self.get_member_feature_count()):
                log_likelihood += gammaln(self.feature_feature_cluster_count_m[x, :, z] + self.beta_m).sum()
                log_likelihood -= gammaln((self.feature_feature_cluster_count_m[x, :, x] + self.beta_m).sum())
                log_likelihood += gammaln(self.feature_feature_cluster_count_j[x, :, z] + self.beta_j).sum()
                log_likelihood -= gammaln((self.feature_feature_cluster_count_j[x, :, z] + self.beta_j).sum())
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







