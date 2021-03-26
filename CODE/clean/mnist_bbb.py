# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:50:05 2021

@author: devin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:06:38 2021

@author: devin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
#from torchsummary import summary
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import torch.nn.utils.prune as prune
import os

from priors import GaussianPrior, GMMPrior
from models import Classifier_BBB
from Hinton import SGDHinton
#%%
#functions for HintonSGD 
def get_lr(epoch, lr0, gamma):
 
    return lr0*gamma**epoch
     
# -----------------------------------------------------------
     
def get_momentum(epoch, p_i, p_f, T):
 
    if epoch<T:
        p = (epoch/T)*p_f + (1 - (epoch/T))*p_i
    else:
        p = p_f
     
    return p
#%%
def density_snr(model):
    device = "cpu"
    model = model.to(device="cpu")
    
    weights =  np.append(model.h1.w.to(device=torch.device(device)).detach().numpy().flatten(), model.h2.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights, model.out.w.to(device=torch.device(device)).detach().numpy().flatten())
    
    #get trained posterior on weights for the 2 hidden layers
    mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w,  model.out.w_mu.detach().numpy().flatten())

    rho_post_w = np.append(model.h1.w_rho.detach().numpy().flatten(), model.h2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.out.w_rho.detach().numpy().flatten())
    #convert rho to sigma
    #sigma_post_w = np.log(1+np.exp(rho_post_w))
    sigma_post_w = np.exp(rho_post_w)
    #calculate SNR = |mu_weight|/sigma_weight
    SNR = abs(mu_post_w)/sigma_post_w
    db_SNR = 10*np.log10(SNR)
    #order the weights by SNR
    sorted_SNR = np.sort(db_SNR)[::-1]
    #remove x% of weights with lowest SNR
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return weights, db_SNR
#%%
input_size = 28*28
hidden_size = 800
output_size = 10
batch_size= 128
path = './MNISTdataset'
#path = './mnist_'
learning_rate = torch.tensor(1e-3) # Initial learning rate {1e-3, 1e-4, 1e-5}
momentum = torch.tensor(9e-1)
burnin = None
T = 1.0
#uncomment for Hinton SGD
'''
learning_rate = torch.tensor(1e-1) #1e-1 # Initial learning rate
p_i = torch.tensor(5e-1) # Initial momentum
p_f = torch.tensor(0.99) #Final momentum
gamma = torch.tensor(0.998) #0.998 #Decay factor
'''
#%% check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)
#%%
#uncomment if the website thinks you're a bot- 'http error 403: forbidden' while downloading MNIST
'''
from six.moves import urllib    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
'''
#%%

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

trainset = datasets.MNIST(root = path, train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,shuffle=True, drop_last=False, **kwargs)

# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets 
# Creating data indices for training and validation splits:
dataset_size = len(trainset)
indices = list(range(dataset_size))
split = 10000 #int(np.floor(validation_split * dataset_size))
shuffle_dataset = True
#random_seed= 42
 
if shuffle_dataset :
    #np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)

 
testset = datasets.MNIST(root = path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,shuffle=True, drop_last=False, **kwargs)


#%%
model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(model.parameters(), lr= torch.tensor(1e-4), weight_decay=torch.tensor(1e-5))
#optimizer = SGDHinton(model.parameters(), lr=learning_rate, momentum=p_i, l2_limit=torch.tensor(0))

#scheduler = ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor=0.7, patience=2, verbose=True)

#%%
num_batches_train = len(train_loader)
num_batches_valid = len(validation_loader)
num_batches_test = len(test_loader)
print(num_batches_train,num_batches_valid, num_batches_test)
#%%
def train():

    train_loss, train_accs=[],[]; acc = 0
    train_loss_c, train_loss_l = [],[]
    #print("training")
    for batch, (x_train, y_train) in enumerate(train_loader):
        model.train()
        x_train, y_train = x_train.to(device), y_train.to(device)
        model.zero_grad()
                
        loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_train, y_train, 1, batch, num_batches_train, samples_batch=len(y_train), T=T, burnin=burnin)
        train_loss.append(loss.item())
        train_loss_c.append(complexity_cost.item())
        train_loss_l.append(likelihood_cost.item())

        acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
        train_accs.append(acc.mean().item())
        
        loss.backward()
        #uncomment for Hinton SGD
        #optimizer.param_groups[0]['lr'] = get_lr(epoch, learning_rate, gamma)
        #optimizer.param_groups[0]['momentum'] = get_momentum(epoch, p_i, p_f, epochs)
        
        optimizer.step()
        
    return train_loss, train_loss_c, train_loss_l, train_accs
#%%
def test(model):
    
    if(epoch==0):
        pass
    else:
        model = Classifier_BBB(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load("model.pt"))
        
    with torch.no_grad():
        
        test_loss, test_accs = [], []; acc = 0
        test_loss_c, test_loss_l = [], []
        for i, (x_test, y_test) in enumerate(validation_loader):
                
            model.eval()
            x_test, y_test = x_test.to(device), y_test.to(device)
            loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_test, y_test, 1, i, num_batches_valid, samples_batch=len(y_test), T=T, burnin=burnin)
            
            acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean()
            
            test_loss.append(loss.item())
            test_loss_c.append(complexity_cost.item())
            test_loss_l.append(likelihood_cost.item())

            test_accs.append(acc.mean().item())
        
        return test_loss, test_loss_c, test_loss_l, test_accs
#%%
#%%

epochs = 1000

epoch_trainaccs, epoch_testaccs = [], []
epoch_trainloss, epoch_testloss = [], []

epoch_trainloss_complexity, epoch_testloss_complexity = [], []
epoch_trainloss_loglike, epoch_testloss_loglike = [], []

epoch_testerr = []

early_stopping = True
_bestacc = 0.


for epoch in range(epochs):

    train_loss, train_loss_c, train_loss_l, train_accs = train()
        
    print('Epoch: {}, Train Loss: {}, Train Accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(train_accs)))
    
    
    test_loss, test_loss_c, test_loss_l, test_accs = test(model)

    print('Epoch: {}, Test Loss: {}, Test Accuracy: {}, Test Error: {}'.format(epoch, np.mean(test_loss), np.mean(test_accs), 100.*(1 - np.mean(test_accs))))
    #print('Epoch: {}, Test Loss: {}, Test Accuracy: {}, Test Error: {}, NLL: {}'.format(epoch, np.mean(test_loss), np.mean(test_accs), 100.*(1 - np.mean(test_accs)), np.mean(test_loss_l)))
    

    #save density and snr
    
    if(epoch % 50 ==0 ): #40
        density, db_SNR = density_snr(model)            
        with open('density' + str(epoch) + '.txt', "wb") as fp:
            pickle.dump(density, fp)
        with open('snr' + str(epoch) + '.txt', "wb") as fp:
            pickle.dump(db_SNR, fp)
    
    epoch_trainaccs.append(np.mean(train_accs))
    epoch_testaccs.append(np.mean(test_accs))
    epoch_testerr.append(100.*(1 - np.mean(test_accs)))
    epoch_trainloss.append(np.mean(train_loss))
    epoch_testloss.append(np.mean(test_loss))
    
    epoch_trainloss_complexity.append(np.sum(train_loss_c)/len(train_sampler))
    epoch_trainloss_loglike.append(np.mean(train_loss_l))

    epoch_testloss_complexity.append(np.sum(test_loss_c)/len(valid_sampler))
    epoch_testloss_loglike.append(np.mean(test_loss_l))
    
    #scheduler.step(epoch_testloss_loglike[-1])
    
    accuracy = epoch_testaccs[-1]
    # check early stopping criteria:
    if early_stopping and accuracy>_bestacc:
        _bestacc = accuracy
        torch.save(model.state_dict(), "model.pt")
        best_acc = accuracy
        best_epoch = epoch

    
print('Finished Training')
print("Final validation error: ",100.*(1 - epoch_testaccs[-1]))

if early_stopping:
    print("Best validation error: ",100.*(1 - best_acc)," @ epoch: "+str(best_epoch))

if not early_stopping: 
    torch.save(model.state_dict(), "model.pt") 
 

#%%
density, db_SNR = density_snr(model)            
'''
with open('density' + str(1000) + '.txt', "wb") as fp:   #Pickle it
    pickle.dump(density, fp)
with open('snr' + str(1000) + '.txt', "wb") as fp:   #Pickle it
    pickle.dump(db_SNR, fp)
'''
#%%
## plots
pl.figure(dpi=200)
pl.plot(epoch_trainloss, label='train loss')
pl.plot(epoch_testloss, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
#pl.ylim(-0.05,0.2) 
#pl.yscale('log')
#pl.axvline(13, linestyle='--', color='g',label='Early Stopping Checkpoint')
pl.xlabel('Epochs')
pl.ylabel('Loss')
pl.show()
#%%
pl.figure(dpi=200)
pl.plot(epoch_trainloss_complexity, label='train loss')
#pl.plot(epoch_testloss_complexity, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('Weighted complexity cost')
pl.show()
#%%
pl.figure(dpi=200)
pl.plot(epoch_trainloss_loglike, label='train loss')
pl.plot(epoch_testloss_loglike, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('negative log likelihood cost')
pl.show()
#%%
pl.figure(dpi=200)
pl.plot(epoch_testerr)
#pl.legend(loc='lower right')
#pl.ylim(1.3,  5)
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('Validation error(%)')
pl.show()
#%%
plt.figure(dpi=200)
plt.xlabel('Weight')
sns.set_palette("colorblind")
sns.kdeplot(density, x="weight", fill=True,alpha=0.5)
#%%
#plot density and CDF of SNR(dB)
plt.figure(dpi=200)
plt.xlabel('Signal-to-Noise')
#plt.xlim((-35,15))
sns.set_style("whitegrid", {'axes.grid' : True})
sns.kdeplot(db_SNR, x="SNR", fill=True,alpha=0.5)
#%%
plt.figure(dpi=200)
plt.ylabel('CDF')
plt.xlabel('Signal-to-Noise')
sns.kdeplot(db_SNR, x="SNR", fill=False,alpha=0.5, cumulative=True, color= 'black')
#%%
'''
name = 'bbb28'

with open("testloss_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_testloss, fp)
with open("trainloss_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_trainloss, fp)
with open("testerr_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_testerr, fp)
    
with open("testloss_complexity_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_testloss_complexity, fp)
with open("testloss_likelihood_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_testloss_loglike, fp)

with open("trainloss_complexity_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_trainloss_complexity, fp)
with open("trainloss_likelihood_"+ name+".txt", "wb") as fp:   #Pickle it
    pickle.dump(epoch_trainloss_loglike, fp)
'''

#%% testing loop

#load model
model = Classifier_BBB(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("model.pt"))
#model = torch.load("model.pt")

with torch.no_grad():
    model.eval()        
    test_loss, test_accs = [], []; acc = 0
    test_loss_c, test_loss_l = [], []
    for i, (x_test, y_test) in enumerate(test_loader):
       
        x_test, y_test = x_test.to(device), y_test.to(device)
            #samples = 5?
        loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_test, y_test, 1, i, num_batches_test,samples_batch=len(y_test), T=T, burnin=burnin)
            
        acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean()
            
        test_loss.append(loss.item())
        test_loss_c.append(complexity_cost.item())
        test_loss_l.append(likelihood_cost.item())

        test_accs.append(acc.mean().item())
        
testaccs= np.mean(test_accs)
testerr = (100.*(1 - np.mean(test_accs)))
testloss = np.mean(test_loss)
#epoch_testloss_complexity.append(np.mean(test_loss_c))
#epoch_testloss_loglike.append(np.mean(test_loss_l))
print("Test error:", testerr)
