import numpy as np
import matplotlib.pyplot as plt
import flif2 as flif
import cv2 as cv

img = cv.imread("spectral_frac_diff/data/trees.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (32, 32))

img = img / sum(img)

img = img.flatten()
num_neurons = 32*32

simulation_time = 100 # simulation_time * dt (0.1 ms) = biological time simulated
input_noise = 0 #0.5
connectivity = 1 #0.28261
inhibitory = 0.5 #0.2
self_connections = False

input_data = [img for _ in range(simulation_time)]
input_data1 = [np.random.rand(28*28) for _ in range(simulation_time)]

input_zeros = [np.zeros(28*28) for _ in range(simulation_time)]

'''
input_length = 100
input_data = np.concatenate((input_data, [np.zeros(28*28) for _ in range(simulation_time - input_length)]))
'''

activity = []
alphas = np.linspace(0.2, 1, 5)
for alpha in [0.1, 0.2, 0.3, 0.4, 0.51976, 0.6, 0.7, 0.8, 1]:
    net = flif.Net(num_neurons, input_noise, connectivity, inhibitory, self_connections)
    activity, weights = net.run(alpha, simulation_time, input_data)
    x = np.linspace(0, simulation_time, simulation_time)
    size, count = np.unique(activity, return_counts=True)
    plt.plot(x, activity, label=alpha)
    #plt.imshow(weights)
plt.legend()
plt.show()
