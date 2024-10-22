import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define network parameters
width = 5
height = 5
num_steps = 10

class Node:
    def __init__(self):
        # Randomly initialize weights for all-to-all connections
        self.weights = np.random.rand(width, height)
        # Initialize neuron state (0 = off, 1 = spiking)
        self.state = 0
        # Set the membrane potential to the resting value (-70 mV)
        self.membrane_potential = -70
        # Threshold for spiking
        self.threshold = -50
        # Trace for fractional LIF
        self.V_trace = [-70]  # Start with resting potential in the trace

    # Standard LIF model
    def lif(self, V, time_step=0.1, I=0, gl=0.0025, Cm=0.5, VL=-70):
        tau = Cm / gl
        V = V + (time_step/tau) * (-1 * (V - VL) + I / gl)
        return V

    # Fractional LIF model
    def flif(self, V_trace, V_weight, I=0, thresh=-50, V_reset=-70, Vl=-70, dt=0.1, beta=0.05, gl=0.0025, Cm=0.5):
        N = len(V_trace)
        V = V_trace[N - 1]
        tau = Cm / gl

        # V_new is the voltage at t_N+1
        # Markov term computation
        V_new = dt**(beta) * math.gamma(2 - beta) * (-gl * (V - Vl) + I) / Cm + V

        # Compute voltage trace
        delta_V = np.subtract(V_trace[1:], V_trace[0:(N - 1)])
        print(len(V_weight[-len(delta_V):]))
        print(len(delta_V))
        memory_V = np.inner(V_weight[-len(delta_V):], delta_V)

        V_new -= memory_V

        # Check for spike and reset voltage if needed
        spike = (V_new > thresh)
        if spike:
            V_new = V_reset  # Reset potential after spike

        return V_new, spike

class Net:
    def __init__(self, prob):
        self.nodes = [[Node() for _ in range(width)] for _ in range(height)]

        # Randomly set initial state (spiking) based on probability
        for i in range(width):
            for j in range(height):
                if prob > np.random.rand():
                    self.nodes[i][j].state = 1
                    self.nodes[i][j].membrane_potential = -50  # Set initial spiking potential

    def get_states(self):
        states = [[x.state for x in row] for row in self.nodes]
        return states

    def get_membrane_potentials(self):
        potentials = [[x.membrane_potential for x in row] for row in self.nodes]
        return potentials

    # Run the network for multiple time steps without external input
    def run(self, num_steps, beta=0.6):
        V_weight = (np.arange(num_steps - 1) + 1) ** (1 - beta) - np.arange(num_steps - 1) ** (1 - beta)
        for step in range(num_steps-1):
            for i in range(width):
                for j in range(height):
                    node = self.nodes[i][j]
                    # First two steps use standard LIF
                    if step < 2:
                        node.V_trace.append(float(node.membrane_potential))
                        V = node.membrane_potential
                        node.membrane_potential = node.lif(V)
                    else:
                        node.V_trace.append(float(node.membrane_potential))
                        node.membrane_potential, spike = node.flif(node.V_trace, V_weight)
                        node.state = 1 if spike else 0
# Function to animate the spiking states over time
def animate_spiking(net, num_steps, interval=200):
    fig, ax = plt.subplots()

    # Initialize the plot
    im = ax.imshow(net.get_states(), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Neuron States (1 = Spiking, 0 = Not Spiking)")

    def update(frame):
        # Run one time step of the simulation
        net.run(3)
        # Update the image with the new states
        im.set_data(net.get_states())
        return [im]

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, repeat=True)
    plt.show()

# Set initial probability for neurons to start "on" (spiking)
prob = 0.5

# Initialize the network
net = Net(prob)

for step in range(num_steps):
    print(np.stack(net.get_states()))
    net.run(10)
# Animate the spiking network over 100 steps
#animate_spiking(net, num_steps=1000, interval=200)  # 200ms per frame (adjust as needed)

