# plotting
import matplotlib.pyplot as plt
from IPython.display import HTML
from snntorch import spikeplot as splt

# plot input current, membrane potential, spikeplot for one lif
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

# plot input current, membrane potential, spikeplot for two lifs
def plot2_cur_mem_spk(cur, mem1, mem2, spk1, spk2, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25, ylim_max3=1.25, ylim_max4=1.25):
    # Generate Plots
    fig, ax = plt.subplots(5, figsize=(8,6), sharex=True, gridspec_kw = {'height_ratios': [1, 1, 1, 0.4, 0.4]})

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([0, ylim_max1])
    ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("I")
    if title:
        ax[0].set_title(title)

    # Plot membrane potential for first lif
    ax[1].plot(mem1)
    ax[1].set_ylim([0, ylim_max2])
    ax[1].set_ylabel("U_1")
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    
    # Plot membrane potential for second lif
    ax[2].plot(mem2)
    ax[2].set_ylim([0, ylim_max3])
    ax[2].set_ylabel("U_2")
    if thr_line:
        ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)


    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk1, ax[3], s=400, c="black", marker="|")
    splt.raster(spk2, ax[4], s=400, c='black', marker="|")

    if vline:
        ax[3].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
        ax[4].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
 

    plt.yticks([])

    plt.show()


