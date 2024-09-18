# Simple example of spike coding mnist data based on documentation from snntorch.readthedocs.io

import torch
import torch.nn as nn
import snntorch as snn

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

from snntorch import utils
from snntorch import spikegen

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

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# Rate coding mnist data

# iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

#spiking data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

#spikeplot animation example

spike_data_sample = spike_data[:, 0, 0]

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

HTML(anim.to_html5_video())

anim.save("../test.mp4")



