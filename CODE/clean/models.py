# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:04:26 2021

@author: devin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from layers import Linear_BBB


class Classifier_BBB(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        #super(Classifier_BBB, self).__init__()
        self.h1  = Linear_BBB(in_dim, hidden_dim)
        self.h2  = Linear_BBB(hidden_dim, hidden_dim)
        self.out = Linear_BBB(hidden_dim, out_dim)
        self.out_dim = out_dim
    
    def forward(self, x):
        x = x.view(-1, 28*28) #flatten
        #x = torch.sigmoid(self.h1(x))
        #x = torch.sigmoid(self.h2(x))
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x
    
    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.out.log_prior
        
    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.out.log_post
    
    def log_like(self,outputs,target):
        #log P(D|w)
        #return F.nll_loss(outputs, target, reduction='sum')
        return F.nll_loss(outputs, target, reduction='mean')
    
    # avg cost function over no. of samples = {1, 2, 5, 10}
    def sample_elbo(self, input, target, samples, batch, num_batches, samples_batch, T=1.0, burnin=None):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        outputs = torch.zeros(samples, target.shape[0], self.out_dim).to(self.device)
        
        log_priors = torch.zeros(samples).to(self.device)
        log_posts = torch.zeros(samples).to(self.device)
        log_likes = torch.zeros(samples).to(self.device)
        
        for i in range(samples):
            outputs[i] = self(input)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = self.log_like(outputs[i,:,:], target)
 
         
        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        
        #log_like = F.nll_loss(outputs.mean(0), target, reduction='sum')
        #print('eblo log_prior', log_prior)
        #print('elbo log_post', log_post)
        #print('elbo log_like', log_like)
        
        if burnin=="blundell":
            frac = 2**(num_batches - (batch + 1))/2**(num_batches - 1)
            #frac = 2**(num_batches-batch+1)/(2**(num_batches) - 1)
        elif burnin==None:
            frac = T/(num_batches*samples_batch) # 1./num_batches #
            
        loss = frac*(log_post - log_prior) + log_like #*num_batches
        '''
        print("IN SAMPLE ELBO")
        print("log_post", log_post)
        print("log_prior", log_prior)
        print("weighted complexity cost", frac*(log_post - log_prior))
        print("likelihood cost", log_like)
        print("loss", loss)
        #print(outputs)
        '''
        complexity_cost = frac*(log_post - log_prior)
        likelihood_cost = log_like
        
        return loss, outputs, complexity_cost, likelihood_cost
