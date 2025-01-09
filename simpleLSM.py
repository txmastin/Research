import numpy as np
import matplotlib.pyplot as plt

class SpikingLiquidStateMachine:
    def __init__(self, 
                 n_reservoir=100, 
                 connectivity=0.2, 
                 spectral_radius=0.95, 
                 input_scaling=1, 
                 leak_rate=0.2, 
                 threshold=0.7, 
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
        self.W_out = np.random.rand(1, n_reservoir)

        # Initialize neuron states
        self.neuron_states = np.zeros(n_reservoir)
        self.neuron_spikes = np.zeros(n_reservoir)
        self.fired = np.zeros(n_reservoir, dtype=bool)
        self.refractory_counters = np.zeros(n_reservoir, dtype=int)
    
        self.neuron_spikes_prev = np.zeros(n_reservoir)

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

def critical(
    lsm, 
    input_current=0.5, 
    max_iterations=1000, 
    tolerance=0.01, 
    learning_rate=0.001
):
    """
    Adjusts the reservoir weights of the liquid state machine to drive it toward criticality.
    Now includes a small constant input current to keep the network active.
    
    Parameters:
        lsm (SpikingLiquidStateMachine): The LSM instance.
        input_current (float): Small constant current added to all neurons.
        max_iterations (int): Maximum number of iterations for weight updates.
        tolerance (float): Target difference between branching ratio and 1.
        learning_rate (float): Scaling factor for weight updates.
        
    Returns:
        None
    """
    def compute_branching_ratio(lsm):
        # Count parent and child spikes
        parent_spikes = np.sum(lsm.neuron_spikes_prev)
        child_spikes = np.sum(lsm.neuron_spikes)
        
        # Avoid division by zero
        if parent_spikes == 0:
            return 0
        
        return child_spikes / parent_spikes

    
    for iteration in range(max_iterations):
        lsm.neuron_spikes_prev = lsm.neuron_spikes.copy()
        lsm.step(input_current)
        # Compute branching ratio
        branching_ratio = compute_branching_ratio(lsm)
        
        # Check for convergence
        if abs(branching_ratio - 1) < tolerance:
            print(f"Criticality achieved: Branching ratio = {branching_ratio:.4f}")
            break
        
        # Update weights based on branching ratio
        if branching_ratio > 1:  # Supercritical
            lsm.W -= learning_rate * lsm.W  # Decrease weights slightly
        elif branching_ratio < 1:  # Subcritical
            lsm.W += learning_rate * lsm.W  # Increase weights slightly
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Branching ratio = {branching_ratio:.4f}")
    else:
        print("Max iterations reached. Branching ratio:", branching_ratio)

    return


def renormalize(lsm, w_th):
    N = lsm.n_reservoir
    merged = True
    print("N:", N)
    while(merged):
        merged = False
        for i in range(N):
            for j in range(i + 1, N):
                if lsm.W[i, j] > w_th:
                    merged = True
                    print(f"Merging neurons {i} and {j} with weight {lsm.W[i, j]}")

                    lsm.neuron_states[i] = (lsm.neuron_states[i] + lsm.neuron_states[j]) / 2

                    for k in range(N):
                        if k != i and k != j:
                            if lsm.W[i, k] * lsm.W[j, k] == 0:
                                lsm.W[i,k] = lsm.W[k,i] = max(lsm.W[i, k], lsm.W[j, k])
                            else:
                                lsm.W[i,k] = lsm.W[k,i] = (lsm.W[i,k] + lsm.W[j,k])/2

                    lsm.W[j, :] = 0
                    lsm.W[:, j] = 0
                    lsm.W_out[:,j] = 0
                    lsm.W_in[j] = 0
                    lsm.neuron_states[j] = 0


def train_output_layer(lsm, input_sequence, target, learning_rate): #target is now a single value
    printing = False
    for value in input_sequence: 
        reservoir_activations, avl = lsm.step(value)
    
    prediction = lsm.predict(reservoir_activations)
    
    error = target - prediction
    
    if printing:
        print("target:", target, "\nprediction", prediction)
        print("error:", error)

    #norm_activations = reservoir_activations / (np.linalg.norm(reservoir_activations) + 1e-6)
    lsm.W_out += learning_rate * np.outer(error, reservoir_activations)
    return abs(error), [avl] #return the absolute error

def test_output_layer(lsm, input_sequence, target): #target is now a single value
    printing = False
    for value in input_sequence:
        reservoir_activations, avl = lsm.step(value) #only take the last value of the input sequence
    prediction = lsm.predict(reservoir_activations)
    
    error = target - prediction
    
    if printing:
        print("target:", target, "\nprediction", prediction)
        print("error:", error)

    return error, [avl] #return the absolute error



def generate_sine_wave(length, amplitude, frequency):
    x = np.linspace(0, 2 * np.pi * frequency * length, length)
    y = amplitude * np.sin(x)
    return y


lsm = SpikingLiquidStateMachine() 

critical(lsm)

x = input()

# find the maximum weight value of the lsm
m = 0
for l in lsm.W:
    if max(l) > m:
        m = max(l)


num_epochs = 1000  # Increased epochs for better observation

# Create a sine wave dataset
sine_wave = generate_sine_wave(500, 1, 1)
plt.plot(sine_wave)
plt.show()

input_window_size = 10


learning_rate = 0.001

# Reduced learning rate
avg_errors = []  # Store avg error in a single list
avls = []

for epoch in range(num_epochs):
    epoch_error = []
    for i in range(len(sine_wave) - input_window_size -1): #reduce range by one
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]  # Correct target: next single value
        err, avl = train_output_layer(lsm, input_sequence, target, learning_rate)
        avls.extend(avl)
        epoch_error.append(err)
    
    avg_errors.append(np.mean(epoch_error))
    print(f"Epoch {epoch+1}/{num_epochs}, Average Error: {np.mean(epoch_error)}")

warmup = 10
for i in range(warmup):
    lsm.step(0)


renormalize(lsm, 0.99*m)

test_runs = 10
test_errors = []

for test in range(test_runs):
    test_error = []
    output = []
    for i in range(len(sine_wave) - input_window_size -1): #reduce range by one
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]  # Correct target: next single value
        err, avl = test_output_layer(lsm, input_sequence, target)
        avls.extend(avl)
        test_error.append(abs(err))
        output.append(target - err)
    test_errors.append(np.mean(test_error))

    print(f"Test {test+1}/{test_runs}, Average Error: {np.mean(test_error)}")
plt.plot(output)
plt.show()
# Plot the error over all training steps
plt.plot(avg_errors)
plt.plot(test_errors)
plt.xlabel("Training Epoch")
plt.ylabel("Average Error")
plt.title("Average Error During Training")
plt.show()

