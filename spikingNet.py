import torch
import torch.nn as nn
import snntorch as snn
import numpy as np

import matplotlib.pyplot as plt


beta = 0.85
threshold = 1

layers = 10
width = 100

time_steps = 100000
spontaneous_prob = 0.03

class SpikingNetwork(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super(SpikingNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        # Define LIF layers
        self.lif_layers = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') for _ in range(num_layers)
        ])
        
        # Fully connected layers
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

spk_count = []

feedback_input = torch.zeros(width)
# Simulate the network
for t in range(time_steps):
    # Generate random spikes based on spontaneous firing probability
    spontaneous_spikes = (torch.rand(width) < spontaneous_prob).float()

    # Run the network with the random spikes
    spikes, mem, feedback_input = net(spontaneous_spikes+feedback_input)
    spk_count.append(torch.stack(spikes).sum().item())

avalanche, count = np.unique(spk_count, return_counts=True)

plt.scatter(avalanche, count, color='blue', alpha=0.7)
plt.title('Neuronal Avalanche Size Distribution')
plt.xlabel('Avalanche Size')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.show()

