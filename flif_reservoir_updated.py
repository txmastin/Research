import numpy as np
import matplotlib.pyplot as plt
import flif

num_neurons = 100
simulation_time = 10000 # simulation_time * dt (0.1 ms) = biological time simulated
input_noise = 0
connectivity = 0.28261
inhibitory = 0.2
self_connections = True

input_data = [[(np.random.rand() < 0.5) for _ in range(num_neurons)] for _ in range(simulation_time)]


alphas = np.linspace(0.1, 0.9, 5)
for alpha in alphas:
    net = flif.Net(num_neurons, input_noise, connectivity, inhibitory, self_connections)

    activity = net.run(alpha, simulation_time, input_data)

    x = np.linspace(0, simulation_time, simulation_time)

    plt.plot(x, activity, label=alpha)
    
plt.legend()
plt.show()
