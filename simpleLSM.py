import numpy as np
import matplotlib.pyplot as plt

class SpikingLiquidStateMachine:
    def __init__(self, 
                 n_reservoir=1000, 
                 connectivity=0.2, 
                 spectral_radius=0.9, 
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

        self.W_out[:][np.random.rand(*self.W_out.shape) > connectivity] = 0
        
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
        return np.dot(self.W_out, reservoir_activations)
        #return (np.tanh(np.dot(self.W_out, reservoir_activations)) + 1) / 2


def critical(
    lsm,
    input_current=0.3,
    max_iterations=1000000,
    tolerance=0.001,
    learning_rate=0.0001,
    avg_window=100
):
    def compute_branching_ratio(lsm):
        # Count parent and child spikes
        parent_spikes = np.sum(lsm.neuron_spikes_prev)
        child_spikes = np.sum(lsm.neuron_spikes)
        
        # Avoid division by zero
        if parent_spikes == 0:
            return 0
        
        return child_spikes / parent_spikes

    avl = []
    branching_ratios = []
    for iteration in range(max_iterations):
        lsm.neuron_spikes_prev = lsm.neuron_spikes.copy()
        lsm.step(input_current)
        # Compute branching ratio
        branching_ratio = compute_branching_ratio(lsm)
        branching_ratios.append(branching_ratio)
        
        # Maintain a rolling average of branching ratio
        if len(branching_ratios) > avg_window:
            branching_ratios.pop(0)
        
        avg_branching_ratio = np.mean(branching_ratios)
        
        # Check for convergence
        if abs(avg_branching_ratio - 1) < tolerance and iteration > avg_window:
            print(f"Criticality achieved: Average branching ratio = {avg_branching_ratio:.4f}")
            '''
            # Continue sampling avalanches
            for _ in range(100000):  
                lsm.step(input_current)
                avl.append(sum(lsm.neuron_spikes))
            
            # Plot avalanche distribution
            a, c = np.unique(avl, return_counts=True)
            plt.figure()
            plt.loglog(a, c, marker='o', linestyle='none')
            plt.xlabel("Avalanche size")
            plt.ylabel("Frequency")
            plt.title("Avalanche size distribution")
            #plt.show()
            '''
            break
        
        # Update weights based on branching ratio
        if avg_branching_ratio > 1:  # Supercritical
            active_neurons = lsm.neuron_spikes > 0
            lsm.W[active_neurons, :] -= learning_rate * lsm.W[active_neurons, :]

            #lsm.W -= learning_rate * lsm.W  # Decrease weights slightly
        elif avg_branching_ratio < 1:  # Subcritical
            active_neurons = lsm.neuron_spikes > 0
            lsm.W[active_neurons, :] += learning_rate * lsm.W[active_neurons, :]
            #lsm.W += learning_rate * lsm.W  # Increase weights slightly
        
        # Print progress
        if iteration % 10 == 0:  # Print every 10 iterations for clarity
            print(f"Iteration {iteration}: Avg branching ratio = {avg_branching_ratio:.4f}")
    else:
        print("Max iterations reached. Branching ratio:", avg_branching_ratio)

    return


def renormalize(lsm, w_th):
    N = lsm.n_reservoir
    count = 0
    merged = True
    print("N:", N)
    while(merged):
        merged = False
        for i in range(N):
            for j in range(i + 1, N):
                if lsm.W[i, j] > w_th:
                    merged = True
                    count += 1
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
    return count

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
    return abs(error), prediction

def test_output_layer(lsm, input_sequence, target): #target is now a single value
    printing = False
    largest = 0
    for value in input_sequence:
        reservoir_activations, avl = lsm.step(value) 
        if avl > largest:
            largest = avl
    
    prediction = lsm.predict(reservoir_activations)
    
    error = target - prediction
    print(largest)

    if printing:
        print("target:", target, "\nprediction", prediction)
        print("error:", error)

    return error, prediction



def generate_sine_wave(length, amplitude, frequency):
    x = np.linspace(0, 1 + 2 * np.pi * frequency * length, length)
    y = (amplitude * np.sin(x) + 1)
    return y


lsm = SpikingLiquidStateMachine() 

critical(lsm)

# find the maximum weight value of the lsm
m = 0
for l in lsm.W:
    if max(l) > m:
        m = max(l)


num_epochs = 200

sine_wave = generate_sine_wave(500, 1, 1)

input_window_size = 5 


learning_rate = 0.001

avg_errors = []  
avls = []

final_out = []

for epoch in range(num_epochs):
    epoch_error = []
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]
        err, out = train_output_layer(lsm, input_sequence, target, learning_rate)
        epoch_error.append(err)
        if epoch == num_epochs - 1:
            final_out.append(out)

    avg_errors.append(np.mean(epoch_error))
    print(f"Epoch {epoch+1}/{num_epochs}, Average Error: {np.mean(epoch_error)}")
plt.figure()
plt.title("pre-renorm")
plt.plot(avg_errors)


lsm = SpikingLiquidStateMachine()

m = 0
for l in lsm.W:
    if max(l) > m:
        m = max(l)


avg_errors = []
avls = []

final_out = []

for epoch in range(num_epochs):
    epoch_error = []
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]
        err, out = train_output_layer(lsm, input_sequence, target, learning_rate)
        epoch_error.append(err)
        if epoch == num_epochs - 1:
            final_out.append(out)

    avg_errors.append(np.mean(epoch_error))
    print(f"Epoch {epoch+1}/{num_epochs}, Average Error: {np.mean(epoch_error)}")

plt.plot(avg_errors)

plt.show()


'''
test_runs = 10
test_errors = []

for test in range(test_runs):
    test_error = []
    output = []
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]
        err, out = test_output_layer(lsm, input_sequence, target)
        test_error.append(abs(err))
        output.append(out)
    test_errors.append(np.mean(test_error))

    print(f"Test pre-renormalization {test+1}/{test_runs}, Average Error: {np.mean(test_error)}")
plt.plot(output)
plt.show()
'''
n_renorm = renormalize(lsm, 0.9*m)
print(n_renorm)
x = input()
final_out = []

for epoch in range(num_epochs):
    epoch_error = []
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]
        err, out = train_output_layer(lsm, input_sequence, target, learning_rate)
        epoch_error.append(err)
        if epoch == num_epochs - 1:
            final_out.append(out)

    avg_errors.append(np.mean(epoch_error))
    print(f"Epoch {epoch+1}/{num_epochs}, Average Error: {np.mean(epoch_error)}")
plt.figure()
plt.title("post-renorm")
plt.plot(final_out)



lsm = SpikingLiquidStateMachine(n_reservoir=(1000-n_renorm))


num_epochs = 200

sine_wave = generate_sine_wave(500, 1, 1)

input_window_size = 5


learning_rate = 0.001

avg_errors = []
avls = []

final_out = []

for epoch in range(num_epochs):
    epoch_error = []
    for i in range(len(sine_wave) - input_window_size):
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]
        err, out = train_output_layer(lsm, input_sequence, target, learning_rate)
        epoch_error.append(err)
        if epoch == num_epochs - 1:
            final_out.append(out)

    avg_errors.append(np.mean(epoch_error))
    print(f"Epoch {epoch+1}/{num_epochs}, Average Error: {np.mean(epoch_error)}")
plt.figure()
plt.title("control")
plt.plot(final_out)



'''
test_runs = 10
test_errors = []

for test in range(test_runs):
    test_error = []
    output = []
    for i in range(len(sine_wave) - input_window_size -1): #reduce range by one
        input_sequence = sine_wave[i:i+input_window_size]
        target = sine_wave[i + input_window_size]  # Correct target: next single value
        err, out = test_output_layer(lsm, input_sequence, target)
        test_error.append(abs(err))
        output.append(out)
    test_errors.append(np.mean(test_error))

    print(f"Test post-renormalization {test+1}/{test_runs}, Average Error: {np.mean(test_error)}")
plt.plot(output)
plt.show()

'''
# Plot the error over all training steps
plt.figure()
plt.plot(avg_errors)
#plt.plot(test_errors)
plt.xlabel("Training Epoch")
plt.ylabel("Average Error")
plt.title("Average Error During Training")
plt.show()

