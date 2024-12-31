import numpy as np
import matplotlib.pyplot as plt

class SpikingLiquidStateMachine:
    def __init__(self, 
                 n_reservoir=1000, 
                 connectivity=0.4, 
                 spectral_radius=1.2, 
                 input_scaling=1, 
                 leak_rate=0.3, 
                 threshold=0.5, 
                 resting_potential=0.0, 
                 refractory_period=2):
        
        self.n_reservoir = n_reservoir
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.threshold = threshold
        self.resting_potential = resting_potential
        self.refractory_period = refractory_period

        # Initialize reservoir weights
        self.W = np.random.rand(n_reservoir, n_reservoir)
        self.W[np.random.rand(*self.W.shape) > connectivity] = 0
        self.W = self.W - np.diag(np.diag(self.W))  # Remove self-connections
        self.W = self.W / np.max(np.abs(np.linalg.eigvals(self.W))) * spectral_radius

        # Initialize input weights
        self.W_in = np.random.rand(n_reservoir) 
        
        # Initialize output weights        
        self.W_out = np.random.rand(n_reservoir)

        # Initialize neuron states
        self.neuron_states = np.zeros(n_reservoir)
        self.neuron_spikes = np.zeros(n_reservoir)
        self.fired = np.zeros(n_reservoir, dtype=bool)
        self.refractory_counters = np.zeros(n_reservoir, dtype=int)
    
    def step(self, input_signal):
        self.neuron_spikes = self.fired.astype(int) 
        total_input = np.dot(self.W, self.neuron_spikes) + self.W_in * input_signal * self.input_scaling

        # Refractory handling: block input accumulation for refractory neurons
        refractory_mask = self.refractory_counters > 0
        total_input[refractory_mask] = 0

        # Update neuron states with leak and input
        self.neuron_states = (1 - self.leak_rate) * self.neuron_states + total_input

        # Detect spiking neurons
        self.fired = self.neuron_states > self.threshold
        self.neuron_states[self.fired] = self.resting_potential

        self.refractory_counters[self.fired] = self.refractory_period

        # Reduce refractory counters
        self.refractory_counters[refractory_mask] -= 1
        return self.neuron_states, sum(self.fired)

    def predict(self, reservoir_activations):
        return np.tanh(np.dot(self.W_out, reservoir_activations))
    

def train_output_layer(slsm, input_sequence, target_sequence, learning_rate):
    error_trace = []
    avls = []
    for input, target in zip(input_sequence, target_sequence):
        reservoir_activations, avl = slsm.step(input)
        prediction = slsm.predict(reservoir_activations)
        print("target:", target, "\nprediction", prediction)
        error = target - prediction
        error_trace.append(error)
        slsm.W_out += learning_rate * error * reservoir_activations
        print("error:", error)
        avls.append(avl)
    return error_trace, avls

def generate_sine_wave(length, amplitude, frequency):
    x = np.linspace(0, 2 * np.pi * frequency * length, length)
    y = amplitude * np.sin(x)
    return y


# Create a sine wave dataset
sine_wave = generate_sine_wave(1000, 1, 0.001)

slsm = SpikingLiquidStateMachine() 

num_epochs = 10
input_window_size = 10
learning_rate = 0.0005
error_trace = []
avls = []

# Train the SLSM
for epoch in range(num_epochs):
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target_sequence = sine_wave[i+1:i+input_window_size+1]
        err, avl = train_output_layer(slsm, input_sequence, target_sequence, learning_rate)
        avls.append(avl)
        error_trace.append(err)

plt.plot(error_trace)
plt.show()

