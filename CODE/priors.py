import torch


class GaussianPrior():

    def __init__(self,var):

        self.var = var

    def log_prob(self,x):

        lnP = torch.distributions.Normal(0,self.var).log_prob(x)

        return lnP

# -----------------------------------------------------------------------------

class GMMPrior():

    def __init__(self,parms):

        if len(parms)!=3:
            print("Incorrect dimensions for prior parameters")
        else:
            self.pi, self.var1, self.var2 = parms

    def log_prob(self,x):

        lnP = self.pi*torch.distributions.Normal(0,self.var1).log_prob(x)
        lnP+= (1-self.pi)*torch.distributions.Normal(0,self.var2).log_prob(x)

        return lnP
