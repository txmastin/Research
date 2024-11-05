import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

time_steps = 100
num_inputs = 10
num_hidden = 10
num_outputs = 10

beta = 0.9

fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta)

mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

mem2_rec = []
spk1_rec = []
spk2_rec = []

spk_in = spikegen.rate_conv(torch.rand((time_steps, num_inputs))).unsqueeze(1)

cur1 = fc1(spk_in[0])
spk1, mem1 = lif1(cur1, mem1)
cur2 = fc2(spk1)
spk2, mem2 = lif2(cur2, mem2)

spk1_rec.append(spk1)
spk2_rec.append(spk2)

for step in range(time_steps):
    cur1 = fc1(spk2)
    spk1, mem1 = lif1(cur1, mem1)
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)
    

mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)


