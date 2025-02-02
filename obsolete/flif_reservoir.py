import numpy as np
import math
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weight_scale):
        # Randomly initialize weights for all-to-all connections
        self.weights = weight_scale*np.random.rand(width, height)
        # Initialize neuron state (0 = off, 1 = spiking)
        self.spike = 0
        # Set the membrane potential to the resting value (-70 mV)
        self.membrane_potential = -70
        # Threshold for spiking
        self.threshold = -50
        # Voltage trace for fractional LIF
        self.V_trace = []
        # Spike trace
        self.spike_trace = []
        
    # Standard LIF model
    # Used for first two simulated time steps
    def lif(self, V, dt=0.1, I=0, gl=0.0025, Cm=0.5, VL=-70):
        tau = Cm / gl
        V = V + (dt/tau) * (-1 * (V - VL) + I / gl)
        return V

    # Fractional LIF model
    def flif(self, V_trace, I=0, thresh=-50, V_reset=-70, Vl=-70, dt=0.1, alpha=0.9, gl=0.0025, Cm=0.5):
        # receive V_trace, set N = length of V_trace - i.e., how many past membrane potentials that have been recorded
        N = len(V_trace)
        V_old = V_trace[N - 1] # previous voltage value at t_N-1
        
        # V_new is the voltage at t_N+1
        # Markov term computation
        markov_term = dt**(alpha) * math.gamma(2 - alpha) * (-gl * (V_old - Vl) + I) / Cm + V_old

        # Compute voltage trace
        V_memory = 0
        for k in range(N-2):
            V_memory += (V_trace[k+1] - V_trace[k]) * ((N-k)**(1-alpha)-(N-1-k)**(1-alpha))
        V_new = markov_term - V_memory
        
        spike = (V_new > thresh)
        if spike:
            V_new = V_reset  # Reset potential after spike

        return V_new, spike

class Net:
    def __init__(self, width, height, weight_scale, input_noise, connectivity, inhibitory, self_connections):
        self.neurons = [[Neuron(weight_scale) for _ in range(width)] for _ in range(height)]
        num_neurons = width*height
        
        # Randomly set initial state (spiking) based on probability
        for i in range(width):
            for j in range(height):
                neuron = self.neurons[i][j]
                if input_noise > np.random.rand():
                    neuron.spike = 1
                    neuron.spike_trace.append(1)
                    neuron.membrane_potential = -50  # Set initial spiking potential
                else:
                    neuron.spike_trace.append(0)
                
                # toggle recurrent connections on/off
                if not self_connections:
                    for k in range(len(neuron.weights)):
                        neuron.weights[i][j] = 0

                # trim synapses based on connectivity
                for l in range(width):
                    for m in range(height):
                        if connectivity < np.random.rand():
                            neuron.weights[l][m] = 0
                        if inhibitory > np.random.rand():
                            neuron.weights[l][m] *= -1

    def get_spikes(self):
        spikes = [[x.spike for x in row] for row in self.neurons]
        return spikes

    def get_membrane_potentials(self):
        potentials = [[x.membrane_potential for x in row] for row in self.nodes]
        return potentials

    # Run the network for multiple time steps without external input
    def run(self, alpha, simulation_time):
        activity = []
        for time_step in range(simulation_time):
            for i in range(width):
                for j in range(height):
                    neuron = self.neurons[i][j]
                    #print("weights:", neuron.weights)
                    #print("spikes:", self.get_spikes())
                    I = float(sum(sum(np.multiply(neuron.weights, self.get_spikes()))))
                    
                    # First two steps use standard LIF
                    if time_step < 2:
                        neuron.V_trace.append(float(neuron.membrane_potential))
                        V = neuron.membrane_potential
                        neuron.membrane_potential = neuron.lif(V, I=I)

                    # Otherwise, use FLIF
                    else:
                        # store old membrane potential in V_trace
                        neuron.V_trace.append(float(neuron.membrane_potential))
                        # retrieve new membrane potential and spike value
                        neuron.membrane_potential, neuron.spike = neuron.flif(neuron.V_trace, I=I, alpha=alpha)
                        # assign
                        #neuron.spike = 1 if spike else 0
                        neuron.spike_trace.append(neuron.spike)
            
            activity.append(float(sum(sum(np.stack(self.get_spikes())))))
            print(f"Alpha:{alpha} {(time_step/simulation_time*100):2.1f}%")
        return activity

# Define network parameters
width = 28
height = 28

simulation_time = 200 # simulation_time * dt (0.1 ms) = biological time simulated

weight_scale = 1 #0.381

# probability a given neuron starts with a 'spike'
input_noise = 0.5 #0.251

# connectivity (percent value [0,1])
#connectivity = 0.205
connectivity = 0.28261

inhibitory = 0.2

self_connections = True



alphas = np.linspace(0.6, 0.825, 10)
for alpha in alphas:
    # Initialize the network
    net = Net(width, height, weight_scale, input_noise, connectivity, inhibitory, self_connections)

    activity = net.run(alpha, simulation_time)

    x = np.linspace(0, simulation_time, simulation_time)

    plt.plot(x, activity, label=alpha)
    
plt.legend()
plt.show()
