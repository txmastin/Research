import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import numpy as np

import matplotlib.pyplot as plt


beta = 0.95
threshold = 1

layers = 10 
width = 10

num_trials = 1 
time_steps = 100000
spontaneous_prob = 0.05

class SpikingNetwork(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super(SpikingNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        # LIF layers
        self.lif_layers = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract') for _ in range(num_layers)
        ])
        
        # FC layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(num_neurons, num_neurons) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        mem = []
        spikes = []
        
        for i in range(self.num_layers):
            if i == 0:
                x_in = x
            else:
                x_in = spikes[-1]  # Input from the previous layer
            
            # Apply the LIF layer
            spike, mem_val = self.lif_layers[i](x_in)
            spikes.append(spike)
            mem.append(mem_val)
            
            # Fully connected layer (except for the last layer)
            if i < self.num_layers - 1:
                x_in = self.fc_layers[i](spike)
            
        # Feedback connection from output layer to input layer (last layer)
        feedback_input = spikes[-1]
        return spikes, mem, feedback_input

# Create the network
net = SpikingNetwork(layers, width)
'''
with torch.no_grad():
    for param in net.parameters():
        param += 1
'''
avalanche_count = []
in_avalanche = False
current_avalanche = 0

feedback_input = torch.zeros(width)

for trial in range(num_trials):
    for t in range(time_steps):

        spontaneous_spikes = (torch.rand(width) < spontaneous_prob).float()
        # rate_coded_spikes = spikegen.rate_conv(torch.rand(width))  

        spikes, mem, feedback_input = net(spontaneous_spikes+feedback_input)
        
        if torch.cat(spikes).sum().item() > 0:
            in_avalanche = True
            current_avalanche += torch.cat(spikes).sum().item()
        elif in_avalanche:
            avalanche_count.append(current_avalanche)
            in_avalanche = False
            current_avalanche = 0
        #print(torch.stack(spikes))
        # Count number of avalanches
    avalanche, count = np.unique(avalanche_count, return_counts=True)

        # Plotting
    plt.scatter(avalanche, count, color='blue', alpha=0.7)

plt.title('Neuronal Avalanche Size Distribution')
plt.xlabel('Avalanche Size')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.show()

