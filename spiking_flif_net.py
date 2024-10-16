import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen

import numpy as np
import matplotlib.pyplot as plt

import flif as fnn

num_steps = time_steps = 100000

spontaneous_prob = 0.5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

avalanche_count = []
in_avalanche = False
current_avalanche = 0

num_input = 80
num_hidden = 80
num_output = 80


#feedback_input = torch.zeros(width)
spk_mem = []
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for alpha in alphas:
    net = fnn.FSNN(num_input, num_hidden, num_output, num_steps, device, alpha)
    for t in range(time_steps):

        spontaneous_spikes = (torch.rand(num_input) < spontaneous_prob).float()
        # rate_coded_spikes = spikegen.rate_conv(torch.rand(width))  

        out_spikes, hid_spikes = net(spontaneous_spikes)
        spikes = torch.stack((out_spikes, hid_spikes))
        if spikes.sum().item() > 0:
            in_avalanche = True
            current_avalanche += spikes.sum().item()
        elif in_avalanche:
            avalanche_count.append(current_avalanche)
            in_avalanche = False
            current_avalanche = 0
        #print(torch.stack(spikes))
        # Count number of avalanches
        if (t % 1000) == 0:
            print(f"alpha = {alpha} \t {int(t/1000)}%")
    avalanche, count = np.unique(avalanche_count, return_counts=True)
    # Plotting
    plt.plot(avalanche, count)

plt.title('Neuronal Avalanche Size Distribution')
plt.xlabel('Avalanche Size')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.show()

