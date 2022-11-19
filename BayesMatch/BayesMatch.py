import numpy as np
from numpy.random import multinomial, randint, choice
from scipy.special import psi
from BayesMatchBase import BayesMatchBase


class BayesMatch(BayesMatchBase):

    def __init__(self, corpus_jobs, corpus_members, K, alpha, burnin, samples, interval, eval_every, beta_m=0.01, beta_j=0.01, beta_c=0.01):
        """
        Create BayesMatch instance.
        """
        super().__init__(corpus_jobs, corpus_members, K, alpha, beta_m=0.01, beta_j=0.01, beta_c=0.01)
        self.burnin = burnin
        self.samples = samples
        self.interval = interval
        self.eval_every = eval_every

    def _optimize_prior_alpha(self):
        """
        Adjust alpha using Minka's fixed-point iteration.
        See Minka (2000): Estimating a Dirichlet distribution and
        https://people.cs.umass.edu/~cxl/cs691bm/lec09.html for details.
        """
        num = 0.0
        denom = 0.0
        for member_idx, _ in enumerate(self.corpus_members):
            num += psi(self.doc_topic_count[doc_idx] + self.alpha) - psi(self.alpha)
            denom += psi(np.sum(self.doc_topic_count[doc_idx] + self.alpha)) - psi(np.sum(self.alpha))
        self.alpha *= num / denom

    def _optimize_prior_beta(self):
        """
        Adjust alpha using Minka's fixed-point iteration.
        See Minka (2000): Estimating a Dirichlet distribution and
        https://people.cs.umass.edu/~cxl/cs691bm/lec09.html for details.
        """
        pass

    def random_init(self):
        """
        Initializes all BayesMatch variables at random.
        """
        # initialize member counts
        for member_idx, mem in enumerate(self.corpus_members):
            for pos, feature_idx in enumerate(mem):
                new_cluster_idx = randint(self.K)
                self.update_member(new_cluster_idx, member_idx, feature_idx, pos, 1)

        # initialize job counts
        for job_idx, job in enumerate(self.corpus_jobs):
            for pos, feature_idx in enumerate(job):
                new_cluster_idx = randint(self.K)
                self.update_job(new_cluster_idx, job_idx, feature_idx, pos, 1)

    def _sample_member(self):
        """
        Samples new cluster assignments for features in all members and updates the current state of the posterior
        """
        for member_idx, mem in enumerate(self.corpus_members):
            if member_idx % 100 == 0:
                print("{} / {}\t\t".format(member_idx, len(self.corpus_members)), end="\r")
            for pos, feature_idx in enumerate(mem):
                # get cluster assignment of current feature from member index and position in member
                cluster_idx = self.get_cluster_assignment_m(member_idx, pos)
                # decrement all corpus statistics by one
                self.update_member(cluster_idx, member_idx, feature_idx, pos, -1)
                # compute full conditional posterior vector
                probs = self.members_full_conditional_posterior(feature_idx)
                # sample new cluster_idx, returns index of vector of all 0s and one 1
                new_cluster_idx = multinomial(1, probs).argmax()
                # increment all corpus statistics by on
                self.update_member(new_cluster_idx, member_idx, feature_idx, pos, 1)

    def _sample_job(self):
        """
        Samples new cluster assignments for features in all jobs and updates current state of posterior.
        """
        for job_idx, job in enumerate(self.corpus_jobs):
            if job_idx % 100 == 0:
                print("{} / {}\t\t".format(job_idx, len(self.corpus_jobs)), end="\r")
            for pos, feature_idx in enumerate(job):
                # get cluster assignment of current feature from member index and position in member
                cluster_idx = self.get_cluster_assignment_j(job_idx, pos)
                # decrement all corpus statistics by one
                self.update_job(cluster_idx, job_idx, feature_idx, pos, -1)
                # compute full conditional posterior vector
                probs = self.jobs_full_conditional_posterior(feature_idx)
                # sample new cluster_idx, returns index of vector of all 0s and one 1
                new_cluster_idx = multinomial(1, probs).argmax()
                # increment all corpus statistics by one
                self.update_job(new_cluster_idx, job_idx, feature_idx, pos, 1)

    def fit(self, optimize_priors=False):
        """
        Fits BayesMatch model using collapsed Gibbs sampling
        :return:
        """
        self.random_init()
        for iteration in range(self.burnin + self.samples):
            # 1) generate member and job samples from the chain
            self._sample_member()
            self._sample_job()
            # 2) optimize priors
            if optimize_priors:
                self._optimize_prior_alpha()
                self._optimize_prior_beta()

            # trace metrics to ensure convergency
            if iteration % self.eval_every == 0:
                self.trace_metrics()
                # print log likelihood and perplexity
                log_likelihood = self.log_likelihood_trace[-1]
                perplexity = self.perplexity_trace[-1]
                if iteration >= self.burnin:
                    print("sampling iteration %i perplexity %.1f likelihood %.1f" % (
                        iteration, perplexity, log_likelihood))
                else:
                    print("burnin iteration %i perplexity %.1f likelihood %.1f" % (
                        iteration, perplexity, log_likelihood))
            else:
                print("iteration %i" % iteration)


if __name__ == '__main__':

    # generate random dataset
    n = 50
    cj = np.random.randint(0, 10, size=1000).reshape(200, 5)
    cm = np.random.randint(5, 15, size=1000).reshape(200, 5)

    # initialize matching algorithm
    match = BayesMatch(
        corpus_jobs=cj,
        corpus_members=cm,
        K=10,
        alpha=np.array([1]),
        burnin=0,
        samples=100,
        interval=1,
        eval_every=1
    )

    match.fit()
    match.plot_metrics()







