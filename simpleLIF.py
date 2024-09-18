import snntorch as snn
from snntorch import spikeplot as splt
import torch


# plotting
import matplotlib.pyplot as plt
from IPython.display import HTML

# plot input current, membrane potential, spikeplot
def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([0, ylim_max1])
    ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("Input Current ()")
    if title:
        ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, ylim_max2]) 
    ax[1].set_ylabel("Membrane Potential ()")
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    plt.ylabel("Output spikes")
    plt.yticks([]) 

    plt.show()





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


