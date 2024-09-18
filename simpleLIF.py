import snntorch as snn
from snntorch import spikeplot as splt
import torch


# plotting
import matplotlib.pyplot as plt
from IPython.display import HTML
from plotLIF import *


lif = snn.Leaky(beta=0.8) # LIF neuron with a decay rate of beta

# setup inputs
num_steps = 200

x = torch.cat((torch.zeros(10), torch.ones(190)*0.21))

mem = torch.zeros(1)

spk = torch.zeros(1)

mem_rec = []

spk_rec = []

# neuron simulation

for step in range(num_steps):
    # lif loop
    spk, mem = lif(x[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(x, mem_rec, spk_rec, thr_line=1, ylim_max1=1.0, title="snn.Leaky Neuron Model")


