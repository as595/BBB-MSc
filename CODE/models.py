import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from layers import Linear_BBB

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1  = nn.Linear(in_dim, hidden_dim)
        self.h2  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                nn.init.normal_(m.weight, 0, 1/np.sqrt(y))
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        
        return x, F.log_softmax(x,dim=1)

# -----------------------------------------------------------------------------

class Classifier_MLPDropout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1  = nn.Linear(in_dim, hidden_dim)
        self.h2  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dr1 = nn.Dropout(p=0.5)
        self.dr2 = nn.Dropout(p=0.2)
        self.out_dim = out_dim

    def forward(self, x):
        
        x = self.dr2(x)
        x = F.relu(self.h1(x))
        x = self.dr1(x)
        x = F.relu(self.h2(x))
        x = self.dr1(x)
        x = self.out(x)
           
        return x, F.log_softmax(x,dim=1)

# -----------------------------------------------------------------------------

class Classifier_BBB(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1  = Linear_BBB(in_dim, hidden_dim)
        self.h2  = Linear_BBB(hidden_dim, hidden_dim)
        self.out = Linear_BBB(hidden_dim, out_dim)
        self.out_dim = out_dim
    
    def forward(self, x):
        #x = x.view(-1, 28*28)
        x = torch.sigmoid(self.h1(x))
        x = torch.sigmoid(self.h2(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x
    
    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.out.log_prior
    
    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.out.log_post
    
    def log_like(self,outputs,target):
        return F.nll_loss(outputs, target, reduction='sum')
    
    def sample_elbo(self, input, target, samples, batch, num_batches, burnin=None):
        
        outputs = torch.zeros(samples, target.shape[0], self.out_dim)
        
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        
        for i in range(samples):
            outputs[i] = self(input)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = self.log_like(outputs[i,:,:],target)
    
        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        
        if burnin=="blundell":
            frac = 2**(num_batches - batch + 1)/2**(num_batches - 1)
        elif burnin==None:
            frac = 1./num_batches
        
        loss = frac*(log_post - log_prior) + log_like
        
        return loss, outputs

# -----------------------------------------------------------------------------

