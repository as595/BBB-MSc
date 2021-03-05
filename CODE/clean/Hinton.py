# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:33:28 2020

@author: devin
"""

import torch
from torch.optim.optimizer import Optimizer, required
 
  
 
class SGDHinton(Optimizer):
    r"""Implements stochastic gradient descent with momentum following the update method described in Hinton+ 2012 https://arxiv.org/pdf/1207.0580.pdf including the renormalisation of input weights for individual neurons if they exceed a specified L2 norm threshold.
 
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        l2_limit (float, optional): L2 threshold per neuron (default: 15; set to 0 to remove)
         
    Example:
        >>> optimizer = SGDHinton(model.parameters(), lr=0.1, momentum=0.9, l2_limit=0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
 
    .. note::
        The implementation of SGD with Momentum differs from
        the standard PyTorch SGD implementation
 
        The update can be written as
 
        .. math::
        \begin{aligned}
            v_{t+1} & = \mu * v_{t} + (1 - \mu) * \text{lr} * g_{t+1}, \\
            p_{t+1} & = p_{t} + v_{t+1},
        \end{aligned}
             
        This is in contrast to the standard PyTorch implementation which employs an update of the form
         
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
 
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
         
 
    """
 
    def __init__(self, params, lr=required, momentum=0, l2_limit=15.):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if l2_limit < 0.0:
            raise ValueError("Invalid L2 threshold value: {}".format(l2_limit))
 
        defaults = dict(lr=lr, momentum=momentum, l2_limit=l2_limit)
        super(SGDHinton, self).__init__(params, defaults)
 
    def __setstate__(self, state):
        super(SGDHinton, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
 
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
 
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            momentum = group['momentum']
            learning_rate = group['lr']
            l2_limit = group['l2_limit']
             
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                renorm = torch.ones(p.size())
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                         
                        alpha = learning_rate*(1 - momentum)
                        buf.mul_(momentum).add_(d_p, alpha=-1.*alpha)
                         
                        p_tmp = torch.clone(p).detach()
                         
                        if l2_limit > 0 and len(p_tmp.size()) > 1:
                            l2_norm = [torch.sqrt(torch.sum(p_tmp[i,:]**2)) for i in range(p_tmp.size(0))]
                             
                            for i in range(p_tmp.size(0)):
                                if l2_norm[i].item()>l2_limit:
                                    renorm[i,:] = l2_limit/l2_norm[i].item()
                                 
                        d_p = buf
                         
                p.add_(d_p, alpha=1.).mul_(renorm)
                 
        return loss