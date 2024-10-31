import numpy as np
import math
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self):
        self.spike = 0
        self.membrane_potential = -70
        self.threshold = -50
        self.V_trace = []
        self.spike_trace = []
        self.V_diff = []

    def lif(self, V, dt=0.1, I=0, gl=0.0025, Cm=0.5, VL=-70):
        tau = Cm / gl
        V = V + (dt/tau) * (-1 * (V - VL) + I / gl)
        return V

    def flif(self, V_trace, V_weight, I=0, thresh=-50, V_reset=-70, Vl=-70, dt=0.1, alpha=0.9, gl=0.0025, Cm=0.5):
        N = len(V_trace)
        V_old = V_trace[N - 1] # previous voltage value at t_N-1
        markov_term = dt**(alpha) * math.gamma(2 - alpha) * (-gl * (V_old - Vl) + I) / Cm + V_old
        V_memory = 0
        
        for k in range(N-2):
            V_memory += (V_trace[k+1] - V_trace[k]) * ((N-k)**(1-alpha)-(N-1-k)**(1-alpha))
        '''        
                self.V_diff.append((V_trace[k+1] - V_trace[k]))
        else:
            self.V_diff.append(V_trace[N-1] - V_trace[N-2])
        '''
 
        #V_memory += np.dot(self.V_diff, V_weight[:(N-2)])
        
        
        V_new = markov_term - V_memory
        

        spike = (V_new > thresh)
        if spike:
            V_new = V_reset  
        return V_new, spike

class Net:
    def __init__(self, num_neurons, input_noise, connectivity, inhibitory, self_connections):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.weights = (np.random.rand(num_neurons, num_neurons))
        for i in range(num_neurons):
            neuron = self.neurons[i]
            if input_noise > np.random.rand():
                neuron.spike = 1
                neuron.spike_trace.append(1)
                neuron.membrane_potential = -50  
            else:
                neuron.spike_trace.append(0)
            
            if not self_connections:
                self.weights[i][i] = 0

        for row in range(num_neurons):
            for col in range(num_neurons):
                if connectivity < np.random.rand():
                    self.weights[col][row] = 0
                if inhibitory > np.random.rand():
                    self.weights[col][row] *= -1
    
    def get_spikes(self):
        spikes = [neuron.spike for neuron in self.neurons]
        return spikes

    def get_membrane_potentials(self):
        potentials = [neuron.membrane_potential for neuron in self.neurons]
        return potentials

    def run(self, alpha, simulation_time):
        activity = []
        N = simulation_time
        V_weight = []
        acc = 0
        for k in range(N-2):
            acc+=((N-k)**(1-alpha)-(N-1-k)**(1-alpha))
            V_weight.append(acc)
        print(V_weight)        
        for time_step in range(simulation_time):
            I = np.dot(self.weights, self.get_spikes()) 
            for i in range(num_neurons):
                neuron = self.neurons[i]
                if time_step < 2:
                    neuron.V_trace.append(float(neuron.membrane_potential))
                    V = neuron.membrane_potential
                    neuron.membrane_potential = neuron.lif(V, I=I[i])

                else:
                    neuron.V_trace.append(float(neuron.membrane_potential))
                    neuron.membrane_potential, neuron.spike = neuron.flif(neuron.V_trace, V_weight, I=I[i], alpha=alpha)
                    neuron.spike_trace.append(neuron.spike)
        
            activity.append(float(sum(np.stack(self.get_spikes()))))
            print(f"Alpha:{alpha} {(time_step/simulation_time*100):2.1f}%")
        return activity 

num_neurons = 784
simulation_time = 200 # simulation_time * dt (0.1 ms) = biological time simulated
input_noise = 0.5
connectivity = 0.28261
inhibitory = 0.2
self_connections = True

alphas = np.linspace(0.6, 0.825, 10)
for alpha in alphas:
    net = Net(num_neurons, input_noise, connectivity, inhibitory, self_connections)

    activity = net.run(alpha, simulation_time)

    x = np.linspace(0, simulation_time, simulation_time)

    plt.plot(x, activity, label=alpha)
    
plt.legend()
plt.show()
