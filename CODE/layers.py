import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from priors import GaussianPrior, GMMPrior

class Linear_BBB(nn.Module):

    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=[1,6,6]):

        super().__init__()

        #set dim
        self.input_features = input_features
        self.output_features = output_features

        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.1, 0.1))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-3,-2))

        #initialize bias params
        self.b_mu =  nn.Parameter(torch.zeros(output_features).uniform_(-0.1, 0.1))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(-3,-2))

        # initialize prior distribution
        self.prior = GMMPrior(prior_var)


    def forward(self, input):
        """
          Optimization process
        """
        #sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        #sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        #record prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        #record variational_posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


