import snntorch as snn
from snntorch import spikegen

import torch
import numpy

# plotting
# import matplotlib.pyplot as plt
# from IPython.display import HTML
from plotLIF import *


beta = 0.8

lif1 = snn.Leaky(beta=beta, threshold=1) # LIF neuron with a decay rate of beta
lif2 = snn.Leaky(beta=beta, threshold=1)

# setup inputs
num_steps = 200

#x = torch.cat((torch.zeros(10), torch.ones(190)*0.25))

#x = torch.from_numpy(numpy.random.rand(num_steps)*0.5)

x = spikegen.rate_conv(torch.from_numpy(numpy.random.rand(num_steps)*0.25))


mem1 = torch.zeros(1)
spk1 = torch.zeros(1)
mem1_rec = []
spk1_rec = []

mem2 = torch.zeros(1)
spk2 = torch.zeros(1)
mem2_rec = []
spk2_rec = []


# neuron simulation

for step in range(num_steps):
    # lif loop
    spk1, mem1 = lif1(x[step], mem1)
    spk2, mem2 = lif2(spk1, mem2)

    mem1_rec.append(mem1)
    spk1_rec.append(spk1)
    mem2_rec.append(mem2)
    spk2_rec.append(spk2)



mem1_rec = torch.stack(mem1_rec)
spk1_rec = torch.stack(spk1_rec)

mem2_rec = torch.stack(mem2_rec)
spk2_rec = torch.stack(spk2_rec)


plot2_cur_mem_spk(x, mem1_rec, mem2_rec, spk1_rec, spk2_rec, thr_line=1, ylim_max1=1.0, title="snn.Leaky Neuron Model")


