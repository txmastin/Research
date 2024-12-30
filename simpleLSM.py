import numpy as np
import matplotlib.pyplot as plt

class SpikingLiquidStateMachine:
    """
    Implementation of a Spiking LiquidStateMachine (SLSM).

    Args:
        n_reservoir (int): Number of neurons in the reservoir.
        connectivity (float): Connectivity of the reservoir.
        spectral_radius (float): Spectral radius of the reservoir weight matrix.
        input_scaling (float): Scaling factor for input signals.
        leak_rate (float): Leaking rate of neuron activations.
        threshold (float): Threshold for neuron firing.
        resting_potential (float): Resting potential of neurons.
        refractory_period (int): Refractory period of neurons.
    """

    def __init__(self, 
                 n_reservoir=1000, 
                 connectivity=0.2, 
                 spectral_radius=0.95, 
                 input_scaling=0.0000000000001, 
                 leak_rate=0.95, 
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
        self.W = np.random.rand(n_reservoir, n_reservoir) * connectivity
        self.W[np.random.rand(*self.W.shape) > connectivity] = 0
        self.W = self.W - np.diag(np.diag(self.W))  # Remove self-connections
        self.W = self.W / np.max(np.abs(np.linalg.eigvals(self.W))) * spectral_radius

        # Initialize input weights
        self.W_in = np.random.rand(n_reservoir) 
        
        # Initialize output weights        
        self.W_out = np.random.rand(n_reservoir)

        # Initialize neuron states
        self.neuron_states = np.zeros(n_reservoir)
        
        self.refractory_counters = np.zeros(n_reservoir, dtype=int)

    def step(self, input_signal):
        """
        Simulate one time step of the SLSM.

        Args:
            input_signal (float): Input signal to the SLSM.

        Returns:
            np.array: Reservoir activations after the time step.
        """

        # Calculate total input to neurons
        total_input = np.dot(self.W, self.neuron_states) + self.W_in * input_signal * self.input_scaling
        # Update neuron states with leak and input
        self.neuron_states = (1 - self.leak_rate) * self.neuron_states + total_input

        # Handle refractory period
        self.refractory_counters[self.refractory_counters > 0] -= 1
        firing_neurons = np.where(self.neuron_states > self.threshold)[0]
        self.neuron_states[firing_neurons] = self.resting_potential
        self.refractory_counters[firing_neurons] = self.refractory_period
        return self.neuron_states
    
    def predict(self, reservoir_activations):
        return np.dot(self.W_out, reservoir_activations)

def train_output_layer(slsm, input_sequence, target_sequence, learning_rate):
    error_trace = []
    for input, target in zip(input_sequence, target_sequence):
        reservoir_activations = slsm.step(input)
        prediction = slsm.predict(reservoir_activations)
        print("target:", target, "\nprediction", prediction)
        error = target - prediction
        error_trace.append(error)
        slsm.W_out += learning_rate * error * reservoir_activations.T
    return error_trace

def generate_sine_wave(length, amplitude, frequency):
    x = np.linspace(0, 2 * np.pi * frequency * length, length)
    y = amplitude * np.sin(x)
    return y


# Create a sine wave dataset
sine_wave = generate_sine_wave(1000, 1, 0.1)

slsm = SpikingLiquidStateMachine() 

num_epochs = 10
input_window_size = 5
learning_rate = 0.1
error_trace = []
# Train the SLSM
for epoch in range(num_epochs):
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target_sequence = sine_wave[i+1:i+input_window_size+1]
        error_trace.append(train_output_layer(slsm, input_sequence, target_sequence, learning_rate))

