import snntorch as snn
from snntorch import spikegen

import torch
import numpy

# plotting
# import matplotlib.pyplot as plt
# from IPython.display import HTML
from plotLIF import *

class leakyNeuron:
    def __init__(self, beta, threshold):
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.mem = torch.zeros(1)
        self.spk = torch.zeros(1)
        self.mem_rec = []
        self.spk_rec = []


beta = 0.8
threshold = 1

#lif1 = snn.Leaky(beta=beta, threshold=1) # LIF neuron with a decay rate of beta
#lif2 = snn.Leaky(beta=beta, threshold=1)

width = 1
depth = 2


net = [[leakyNeuron(beta, threshold) for i in range(depth)] for j in range(width)]

lif1 = net[0][0]
lif2 = net[0][1]

# setup inputs
num_steps = 200

#x = torch.cat((torch.zeros(10), torch.ones(190)*0.25))

#x = torch.from_numpy(numpy.random.rand(num_steps)*0.5)

x = spikegen.rate_conv(torch.from_numpy(numpy.random.rand(num_steps)*0.25))

'''
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


for step in range(num_steps):
    lif1.spk, lif1.mem = lif1.lif(x[step], lif1.mem)
    lif2.spk, lif2.mem = lif2.lif(lif1.spk, lif2.mem)
    
    lif1.mem_rec.append(lif1.mem)
    lif1.spk_rec.append(lif1.spk)
    lif2.mem_rec.append(lif2.mem)
    lif2.spk_rec.append(lif2.spk)
'''
for step in range(num_steps):
    for layer in net:
        layer[0].spk, layer[0].mem = layer[0].lif(x[step], layer[0].mem)
        layer[1].spk, layer[1].mem = layer[1].lif(layer[0].spk, layer[1].mem)
        for neuron in layer:
            neuron.mem_rec.append(neuron.mem)
            neuron.spk_rec.append(neuron.spk)
            
mem1_rec = torch.stack(net[0][0].mem_rec)
spk1_rec = torch.stack(net[0][0].spk_rec)

mem2_rec = torch.stack(net[0][1].mem_rec)
spk2_rec = torch.stack(net[0][1].spk_rec)

print(spk1_rec)
print(spk2_rec)

plot2_cur_mem_spk(x, mem1_rec, mem2_rec, spk1_rec, spk2_rec, thr_line=1, ylim_max1=1.0, title="snn.Leaky Neuron Model")


