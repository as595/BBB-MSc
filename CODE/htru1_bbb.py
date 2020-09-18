import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from torchsummary import summary

import numpy as np
import pandas as pd

from models import Classifier_BBB

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

batch_size    = 128                 # number of samples per batch
input_size    = 8                   # The number of features
hidden_size   = 400                 # The number of nodes at the hidden layer
num_classes   = 2                   # The number of output classes. In this case, from 0 to 9
burnin        = None
learning_rate = torch.tensor(1e-4)  # The speed of convergence
momentum      = torch.tensor(9e-1)  # momentum for optimizer
path          = "./datasets/"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# check if a GPU is available:

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)

# -----------------------------------------------------------------------------
# read HTRU1 data:

infile = path+'pulsar.csv.gz'
htru1_df = pd.read_csv(infile, compression='gzip')

htru_x = htru1_df.drop('class', axis=1).values
htru_y = htru1_df['class'].values

# -----------------------------------------------------------------------------
# create pytorch data loaders:

htru_x  = torch.tensor(htru_x, dtype=torch.float)
htru_y  = torch.tensor(htru_y, dtype=torch.long)

x, y = Variable(htru_x), Variable(htru_y)
full_dataset = Data.TensorDataset(x, y)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

valid_loader = Data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               shuffle=True)

num_batches_train = train_size/batch_size
num_batches_test  = test_size/batch_size

# -----------------------------------------------------------------------------
# specify model and optimizer:

model = Classifier_BBB(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# -----------------------------------------------------------------------------
# view a summary of the model:

summary(model, (1,input_size))
model = model.to(device)

# -----------------------------------------------------------------------------

epochs = 100

epoch_trainaccs, epoch_testaccs = [], []
epoch_trainloss, epoch_testloss = [], []

for epoch in range(epochs):  # loop over the dataset multiple times

    train_loss, train_accs=[],[]; acc = 0
    for batch, (x_train, y_train) in enumerate(train_loader):
        
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        model.zero_grad()
        loss, pred = model.sample_elbo(x_train, y_train, 1, batch, num_batches_train, burnin=burnin)
        train_loss.append(loss.item())
        
        acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
        train_accs.append(acc.mean().item())
        
        loss.backward()
        optimizer.step()

    print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(train_accs)))

    with torch.no_grad():
        
        test_loss, test_accs = [], []; acc = 0
        for i, (x_test, y_test) in enumerate(valid_loader):
            
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            loss, pred = model.sample_elbo(x_test, y_test, 5, i, num_batches_test)
            
            acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean()
            
            test_loss.append(loss.item())
            test_accs.append(acc.mean().item())

    print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(test_loss), np.mean(test_accs)))

    epoch_trainaccs.append(np.mean(train_accs))
    epoch_testaccs.append(np.mean(test_accs))

    epoch_trainloss.append(np.mean(train_loss))
    epoch_testloss.append(np.mean(test_loss))

print('Finished Training')
print("Final test error: ",100.*(1 - epoch_testaccs[-1]))

import pylab as pl
pl.subplot(111)
pl.plot(epoch_trainloss)
pl.plot(epoch_testloss)
pl.show()
