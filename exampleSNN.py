# Simple example from snntorch.readthedocs.io

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import utils

# necessary for mnist data

from torchvision import datasets, transforms

# Training Parameters
batch_size = 128
data_path = '/tmp/data/mnist'
num_classes = 10 # i.e., 0-9

# Torch Variables
dtype = torch.float

alpha = 0.9
beta = 0.85

num_steps = 100


# Download mnist dataset

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0), (1))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

# Create a subset of training data 
subset = 10 # reduction factor (i.e., 10 -> reduction of 10x)
mnist_train = utils.data_subset(mnist_train, subset)


# Define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden) # fully connected layer, num_inputs -> num_hidden
        self.lif1 = snn.Leaky(beta=beta) # leaky integrate and fire layer
        self.fc2 = nn.Linear(num_hidden, num_outputs) # fully connected layer, num_hidden -> num_outputs
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky() # initialize membrane potentials of lif1
        mem2 = self.lif2.init_leaky() # initialize membrane potentials of lif2

        spk2_rec = [] # record the output trace of spikes
        mem2_rec = [] # record the output trace of membrance potential

        for step in range(num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk2_rec), torch.stack(mem2_rec)

net = Net().to(device)

output, mem_rec = net(data)

print(output, mem_rec)


            

