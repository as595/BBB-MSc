# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:05:33 2021

@author: devin
"""

import torch
import numpy as np

class GaussianPrior():

    def __init__(self,var):

        self.var = var

    def log_prob(self,x):

        lnP = torch.distributions.Normal(0,self.var).log_prob(x)
 
        return lnP

# -----------------------------------------------------------------------------
#log P(w) = prior on the weights/biases
class GMMPrior():

    def __init__(self,parms):

        if len(parms)!=3:
            print("Incorrect dimensions for prior parameters")
        else:
            self.pi, self.stddev1, self.stddev2 = parms

    def log_prob(self,x):

        var1 = torch.pow(self.stddev1,2)
        var2 = torch.pow(self.stddev2,2)
        prob = (torch.exp(-torch.pow((x-0),2)/(2*var1))/(self.stddev1*np.sqrt(2*np.pi)))*self.pi
        prob+= (torch.exp(-torch.pow((x-0),2)/(2*var2))/(self.stddev2*np.sqrt(2*np.pi)))*(1-self.pi)
        
        logprob = torch.log(prob)

        return logprob