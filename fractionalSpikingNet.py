import torch
import torch.nn as nn
import numpy as np

class FLIFNeuron(nn.Module):
    def __init__(self, size, num_steps, alpha=0.2, dt=1, threshold=-50, V_init=-70, VL=-70, V_reset=-70, gl=0.025, Cm=0.5):
        super(FLIFNeuron, self).__init__()
        self.size = size
        self.num_steps = num_steps
        self.alpha = alpha
        self.dt = dt
        self.threshold = threshold
        self.V_init = V_init
        self.VL = VL
        self.V_reset = V_reset
        self.gl = gl
        self.Cm = Cm
        self.N = 0
        self.delta_trace = torch.zeros(size, num_steps)
        self.memory = torch.zeros(size).float()
        
        # Precompute weights for memory trace
        self.weights = self.compute_weights()

    def compute_weights(self):
        x = self.num_steps
        nv = np.arange(x - 1)
        return torch.tensor((x + 1 - nv) ** (1 - self.alpha) - (x - nv) ** (1 - self.alpha)).float()

    def forward(self, I):
        if self.N == 0:
            V_new = torch.ones(self.size) * self.V_init
            spike = torch.zeros(self.size)
            self.N += 1
        else:
            tau = self.Cm / self.gl
            V_new = V_old + (self.dt / tau) * (-1 * (V_old - self.VL) + I / self.gl)
            V_new += self.dt ** self.alpha * torch.gamma(2 - self.alpha) * (-self.gl * (V_old - self.VL) + I) / self.Cm
            memory_V = torch.matmul(self.delta_trace[:, :self.N-1], self.weights[-self.N+1:])
            V_new -= memory_V

        spike = (V_new > self.threshold).float()
        V_new[spike > 0] = self.V_reset  # Reset voltage on spike
        self.update_memory(V_new)
        
        return spike, V_new

    def update_memory(self, V_new):
        delta = V_new - self.memory.detach()
        self.delta_trace[:, self.N-1] = delta
        self.memory = V_new
        self.N += 1

    def reset(self):
        self.N = 0
        self.delta_trace = torch.zeros(self.size, self.num_steps)

# Define the network
class SpikingNetwork(nn.Module):
    def __init__(self, num_layers, layer_size, num_steps):
        super(SpikingNetwork, self).__init__()
        self.layers = nn.ModuleList([FLIFNeuron(layer_size, num_steps) for _ in range(num_layers)])
        self.layer_size = layer_size

    def forward(self, x):
        for layer in self.layers:
            spikes, x = layer(x)
        return spikes

# Avalanche Detection
def detect_avalanches(spikes):
    
    pass

# Simulation Parameters
num_layers = 3
layer_size = 10
num_steps = 100
network = SpikingNetwork(num_layers, layer_size, num_steps)

# Example Input (random spikes)
input_spikes = torch.randn(layer_size)  # Replace with actual spike data
print(input_spikes)
# Forward pass through the network
spikes = network(input_spikes)
print(spikes)
# Detect avalanches
detect_avalanches(spikes)

