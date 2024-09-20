# Simple example of SNN based on documentation from snntorch.readthedocs.io
 
import torch
import torch.nn as nn
import snntorch as snn

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

from snntorch import utils
from snntorch import spikegen
import itertools
import numpy as np


# necessary for mnist data
from torchvision import datasets, transforms

# Training Parameters
batch_size = 128
data_path = '/tmp/data/mnist'
num_classes = 10 # i.e., 0-9

# Torch Variables
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# temporal dynamics
beta = 0.95
num_steps = 25

# network architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Download mnist dataset

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0), (1))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        
        self.fc1 = nn.Linear(num_inputs, num_hidden) # fully connected layer, num_inputs -> num_hidden
        self.lif1 = snn.Leaky(beta=beta) # leaky integrate and fire layer (beta=beta, threshold=1 (assignable))
        self.fc2 = nn.Linear(num_hidden, num_outputs) # fully connected layer, num_hidden -> num_outputs
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky() # initialize membrane potentials of lif1
        mem2 = self.lif2.init_leaky() # initialize membrane potentials of lif2

        spk2_rec = [] # record the output trace of spikes
        mem2_rec = [] # record the output trace of membrance potential

        # time loop
        for step in range(num_steps):
            cur1 = self.fc1(x.flatten(1)) # flatten(1) since first dimension is batch_size -> batch_size x 784
            spk1, mem1 = self.lif1(cur1, mem1) #spk1 will be 1 if mem1 is above threshold, otherwise 0
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2) #same as spk2
            
            # append latest value to recs
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0) # time-steps x batch x num_out

# load network to device

net = Net().to(device)

# training network

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1 # 60000 / 128 (mnist / batch)
counter = 0

for epoch in range(num_epochs):
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)
        
        # forward pass
        net.train()
        spk_rec, mem_rec = net(data)
        
        # init loss and sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        loss_val = loss(spk_rec.sum(0), targets)
        
        # grad calc and weight updates
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if counter % 10 == 0:
            print(f"Iteration: {counter} \t Train loss: {loss_val.item()}")
        counter+=1
       
