# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:02:00 2021

@author: devin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from priors import GaussianPrior, GMMPrior


#BBB Layer
class Linear_BBB(nn.Module):

    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var= [torch.tensor(3/4), torch.tensor(np.exp(-1)), torch.tensor(np.exp(-7)) ]):
        super().__init__()

        #set dim
        self.input_features = input_features
        self.output_features = output_features
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.1, 0.1).to(self.device))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-5, -4).to(self.device))

        #initialize bias params
        self.b_mu =  nn.Parameter(torch.zeros(output_features).uniform_(-0.1, 0.1).to(self.device))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(-5, -4).to(self.device))
        #self.b = nn.Parameter(torch.zeros(output_features))
        '''
        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).normal_(0, 0.1).to(self.device))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).normal_(-5,0.1).to(self.device))

        #initialize bias params
        self.b_mu =  nn.Parameter(torch.zeros(output_features).normal_(0, 0.1).to(self.device))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(-5,0.1).to(self.device))
        '''
        # initialize prior distribution
        #self.prior = GMMPrior(prior_var)
        self.prior = GaussianPrior(np.exp(0))
        
    def forward(self, input):
        """
          Optimization process
        """

        #sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).to(self.device)
        self.w = self.w_mu + torch.exp(self.w_rho) * w_epsilon
        #print('w',self.w)
        
        #sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(self.device)
        self.b = self.b_mu + torch.exp(self.b_rho) * b_epsilon
        #self.b = nn.Parameter.constant_(0)
        #print('b',self.b)
        
        #record prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior) 
       
        #record variational_posterior - log q(w|theta)
        self.w_post = Normal(self.w_mu.data, torch.exp(self.w_rho))
        self.b_post = Normal(self.b_mu.data, torch.exp(self.b_rho))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        
        '''
        print("IN LINEAR LAYER")
        print('log prior',self.log_prior)
        print('log post', self.log_post)
        '''
        
        return F.linear(input, self.w, self.b)
