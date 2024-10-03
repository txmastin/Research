import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import numpy as np

import matplotlib.pyplot as plt


beta = 0.90
threshold = 1


layers = 10 
width = 10 

num_trials = 1 
time_steps = 10000
spontaneous_prob = 0.15

class SpikingNetwork(nn.Module):
    def __init__(self, num_layers, num_neurons, lr):
        super(SpikingNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.lr = lr

        # LIF layers
        self.lif_layers = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract') for _ in range(num_layers)
        ])
        
        # FC layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(num_neurons, num_neurons) for _ in range(num_layers)
        ])
    
    def stdp_update(self, pre_spikes, post_spikes, weights):
        delta_t = post_spikes - pre_spikes
        
        # long term potentiation (aka reinforcement)
        ltp_mask = (delta_t > 0).float()
        weights += self.lr*ltp_mask
        
        # long term depression
        ltd_mask = (delta_t < 0).float()
        weights -= self.lr*ltd_mask
        print(weights)
        return weights

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
            self.fc_layers[i].weight.data = self.stdp_update(x_in, spike, self.fc_layers[i].weight.data)
            
        # Feedback connection from output layer to input layer (last layer)
        feedback_input = spikes[-1]
        return spikes, mem, feedback_input

# Create the network


lrate = [0.005]
for lr in lrate:
    net = SpikingNetwork(layers, width, lr)
    '''
    with torch.no_grad():
        for param in net.parameters():
            param += 1
    '''
    avalanche_count = []
    in_avalanche = False
    current_avalanche = 0
    avalanche_length = 0
    avalanche_length_count = []
    feedback_input = torch.zeros(width)
    for t in range(time_steps):
        if t % 1000 == 0:
            print(f"Step: {t}")
        spontaneous_spikes = (torch.rand(width) < spontaneous_prob).float()
        # rate_coded_spikes = spikegen.rate_conv(torch.rand(width))  

        spikes, mem, feedback_input = net(spontaneous_spikes+feedback_input)
        
        if torch.cat(spikes).sum().item() > 0:
            in_avalanche = True
            avalanche_length += 1
            current_avalanche += torch.cat(spikes).sum().item()
        elif in_avalanche:
            avalanche_count.append(current_avalanche)
            avalanche_length_count.append(avalanche_length)
            in_avalanche = False
            current_avalanche = 0
            avalanche_length = 0
        #print(torch.stack(spikes))
        # Count number of avalanches
    avalanche, count = np.unique(avalanche_count, return_counts=True)
    length, len_count = np.unique(avalanche_length_count, return_counts=True)

    plt.scatter(avalanche, count, color="blue")
    plt.scatter(length, len_count, color="green")

plt.title('Neuronal Avalanche Size Distribution')
plt.xlabel('Avalanche Size')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.show()

