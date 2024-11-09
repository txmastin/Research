import numpy as np
import matplotlib.pyplot as plt
import flif
import cv2 as cv

img = cv.imread("../playground/bass.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (28, 28))

img = img / sum(img)

img = img.flatten()
print(img)
num_neurons = 28*28

simulation_time = 10000 # simulation_time * dt (0.1 ms) = biological time simulated
input_noise = 0
connectivity = 1 #0.28261
inhibitory = 0.5 #0.2
self_connections = True

input_data = [img for _ in range(simulation_time)]
input_data1 = [np.random.rand(28*28) for _ in range(simulation_time)]


'''
input_length = 100
input_data = np.concatenate((input_data, [np.zeros(28*28) for _ in range(simulation_time - input_length)]))
'''

activity = []
alphas = np.linspace(0.2, 1, 5)
for alpha in alphas:
    net = flif.Net(num_neurons, input_noise, connectivity, inhibitory, self_connections)
    activity = net.run(alpha, simulation_time, input_data1)
    x = np.linspace(0, simulation_time, simulation_time)
    size, count = np.unique(activity, return_counts=True)
    plt.loglog(size, count, label=alpha)
plt.legend()
plt.show()
