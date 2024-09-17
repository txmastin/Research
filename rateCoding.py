import torch
import matplotlib.pyplot as plt
import numpy as np

epoch = 200

num_steps = 1000

raw_vector = torch.ones(num_steps)*0.5

avg_over_time = []

for step in range(epoch):
    rate_coded_vector = torch.bernoulli(raw_vector)
    avg = rate_coded_vector.sum() / len(rate_coded_vector)
    avg_over_time.append(avg.item())
   
x = np.arange(epoch)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

plt.plot(x, avg_over_time)

plt.show()

