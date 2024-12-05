import numpy as np
import matplotlib.pyplot as plt

raw_avg = np.loadtxt('raw_avg.dat', delimiter=',')
raw_diff = np.loadtxt('raw_diff.dat', delimiter=',')

raw_hpf_avg = np.loadtxt('raw_hpf_avg_arr.dat', delimiter=',')
avg = 56.821
labels = ["Raw CIFAR-10", "Spectral Residual", "HPF (Laplacian)"]
data = [avg, avg+raw_avg[-1], avg+raw_hpf_avg[-1]]
plt.bar(labels, data)
#plt.plot(raw_hpf_diff, label="HPF")
#plt.legend()
plt.title("Test Accuracy (Precision)")
plt.show()
