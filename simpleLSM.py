import numpy as np
#import matplotlib.pyplot as plt
import copy

class SpikingLiquidStateMachine:
    def __init__(self, 
                 n_reservoir=1000, 
                 connectivity=0.2, 
                 spectral_radius=0.9, 
                 input_scaling=0.115, 
                 leak_rate=0.2, 
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
    color,
    linestyle,
    label,
    input_current=1,
    max_iterations=50000,
    tolerance=0.000001,
    learning_rate=0.005,
    avg_window=20,
    avalanches=True,
    alter=False
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
    if avalanches:
        # Continue sampling avalanches
        for _ in range(50000):
            lsm.step(input_current)
            avl.append(sum(lsm.neuron_spikes))

        # Plot avalanche distribution
        a, c = np.unique(avl, return_counts=True)
        #plt.loglog(a, c, color=color, linestyle=linestyle, label=label)
        #plt.legend()
        #plt.xlabel("Avalanche size", fontsize=18)
        #plt.ylabel("Frequency", fontsize=18)
        #plt.title("Avalanche size distribution")
        #plt.show()
    
    avl = []
    
    if not alter:
        return

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
        if iteration == max_iterations-1:
            print(f"Average branching ratio = {avg_branching_ratio:.4f}")
            # Continue sampling avalanches
            for _ in range(10000):  
                lsm.step(input_current)
                avl.append(sum(lsm.neuron_spikes))
            # Plot avalanche distribution
            a, c = np.unique(avl, return_counts=True)
            #plt.figure()
            #plt.loglog(a, c, marker='o', linestyle='none')
            #plt.xlabel("Avalanche size")
            #plt.ylabel("Frequency")
            #plt.title("Avalanche size distribution")
            #plt.show()
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
    #print("N:", N)
    while(merged):
        merged = False
        for i in range(N):
            for j in range(i + 1, N):
                if lsm.W[i, j] > w_th:
                    merged = True
                    count += 1
                    #print(f"Merging neurons {i} and {j} with weight {lsm.W[i, j]}")

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

def random_prune(lsm, percent):
    N = lsm.n_reservoir
    count = 0 
    for i in range(N):
        if np.random.rand() < percent:
            count += 1
            lsm.W[i, :] = 0
            lsm.W[:, i] = 0
            lsm.W_out[:,i] = 0
            lsm.W_in[i] = 0
            lsm.neuron_states[i] = 0
    return count

def train_output_layer(lsm, input_sequence, target, learning_rate):
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


def training_loop(lsm, num_epochs, input_window_size, inp, learning_rate):
    avg_errors = []
    final_out = []
    for epoch in range(num_epochs):
        epoch_error = []
        for i in range(len(inp) - input_window_size):
            input_sequence = inp[i:i+input_window_size]
            target = inp[i + input_window_size]
            err, out = train_output_layer(lsm, input_sequence, target, learning_rate)
            epoch_error.append(err)
            if epoch == num_epochs - 1:
                final_out.append(out)

        avg_errors.append(np.mean(epoch_error))
        if(epoch % 100 == 0):
            print(f"Training Step: {epoch}/{num_epochs}, Average Error: {np.mean(epoch_error)}")
    return avg_errors, final_out


with open("datasets/MG/mgdata.dat.txt", 'r') as file:
    lines = file.readlines()



sine_wave = generate_sine_wave(100, 1, 1)
mg = [float(line.split()[1]) for line in lines]

N = 1000

datasets = [sine_wave]

first = True
for data in datasets:
    avg_perf_critical = 0
    avg_perf_control1 = 0
    avg_perf_control2 = 0

    avg_perf_critical_renorm = 0
    avg_perf_control1_renorm = 0
    avg_perf_control2_renorm = 0

    avg_perf_critical_prune = 0
    avg_perf_control1_prune = 0
    avg_perf_control2_prune = 0

    avg_size_critical_renorm = 0
    avg_size_control1_renorm = 0
    avg_size_control2_renorm = 0

    avg_size_critical_prune = 0
    avg_size_control1_prune = 0
    avg_size_control2_prune = 0


    trials = 20

    for i in range(trials):
        # set up reservoirs and copies for comparison

        lsm_critical = SpikingLiquidStateMachine(input_scaling=0.105) 
        lsm_critical_copy = copy.deepcopy(lsm_critical)

        lsm_control1 = SpikingLiquidStateMachine(input_scaling=0.505)
        lsm_control1_copy = copy.deepcopy(lsm_control1)

        lsm_control2 = SpikingLiquidStateMachine(input_scaling=0.22)
        lsm_control2_copy = copy.deepcopy(lsm_control2)
        
        # set up experiment parameters

        num_epochs = 8000
        input_window_size = 5 
        learning_rate = 0.001

        # train original reservoirs and plot performance

        avg_errors_critical, _ = training_loop(lsm_critical, num_epochs, input_window_size, data, learning_rate)


        #plt.figure()
        #plt.plot(avg_errors_critical, color="red", label="Critical Spiking")
        avg_errors_control1, _ = training_loop(lsm_control1, num_epochs, input_window_size, data, learning_rate)
        #plt.plot(avg_errors_control1, color="blue", label="Synchronous Spiking")
        avg_errors_control2, _ = training_loop(lsm_control2, num_epochs, input_window_size, data, learning_rate)
        #plt.plot(avg_errors_control2, color="black", label="Irregular Spiking")
        #plt.xlabel("Training Step", fontsize=18)
        #plt.ylabel("Average Error", fontsize=18)
        #plt.legend()
        #plt.show()
        
        if first:
            avg_errors_critical_str = "data/rg_paper/sine_wave/avg_errors_critical_" + str(i+20) + ".dat"
            np.savetxt(avg_errors_critical_str, avg_errors_critical, delimiter=",")
            avg_errors_control1_str = "data/rg_paper/sine_wave/avg_errors_control1_" + str(i+20) + ".dat"
            np.savetxt(avg_errors_control1_str, avg_errors_control1, delimiter=",")
            avg_errors_control2_str = "data/rg_paper/sine_wave/avg_errors_control2_" + str(i+20) + ".dat"
            np.savetxt(avg_errors_control2_str, avg_errors_control2, delimiter=",")
         
        else:
            avg_errors_critical_str = "data/rg_paper/mg/avg_errors_critical_" + str(i) + ".dat"
            np.savetxt(avg_errors_critical_str, avg_errors_critical, delimiter=",")
            avg_errors_control1_str = "data/rg_paper/mg/avg_errors_control1_" + str(i) + ".dat"
            np.savetxt(avg_errors_control1_str, avg_errors_control1, delimiter=",")
            avg_errors_control2_str = "data/rg_paper/mg/avg_errors_control2_" + str(i) + ".dat"
            np.savetxt(avg_errors_control2_str, avg_errors_control2, delimiter=",")        
        # find the maximum weight value of the lsms and perform renormalization

        m = 0
        for l in lsm_critical.W:
            if max(l) > m:
                m = max(l)

        n_renorm_critical = renormalize(lsm_critical, 0.992*m)

        m = 0
        for l in lsm_control1.W:
            if max(l) > m:
                m = max(l)

        n_renorm_control1 = renormalize(lsm_control1, 0.992*m)

        # find the maximum weight value of the lsm
        m = 0
        for l in lsm_control2.W:
            if max(l) > m:
                m = max(l)

        n_renorm_control2 = renormalize(lsm_control2, 0.992*m)

        percent = ((n_renorm_critical + n_renorm_control1 + n_renorm_control2) / 3) / 1000
        print(percent)

        # random prune the copies for comparison

        #n_prune_critical = random_prune(lsm_critical_copy, percent)
        #n_prune_control1 = random_prune(lsm_control1_copy, percent)
        #n_prune_control2 = random_prune(lsm_control2_copy, percent)

        # plot spike distributions for renormalized and pruned reservoirs
        
        '''
        
        critical(lsm_critical)
        critical(lsm_control1)
        critical(lsm_control2)
        critical(lsm_critical_copy)
        critical(lsm_control1_copy)
        critical(lsm_control2_copy)
        
        '''


        #plt.show()




        # train renormalized reservoirs and plot performance

        avg_errors_critical_renorm, _ = training_loop(lsm_critical, num_epochs, input_window_size, data, learning_rate)
        #plt.figure()
        #plt.plot(avg_errors_critical_renorm, color="red", label="Critical Spiking")

        avg_errors_control1_renorm, _ = training_loop(lsm_control1, num_epochs, input_window_size, data, learning_rate)

        #plt.plot(avg_errors_control1_renorm, color="blue", label="Synchronous Spiking")

        avg_errors_control2_renorm, _ = training_loop(lsm_control2, num_epochs, input_window_size, data, learning_rate)

        #plt.plot(avg_errors_control2_renorm, color="black", label="Irregular Spiking")

        #plt.xlabel("Training Step", fontsize=18)
        #plt.ylabel("Average Error", fontsize=18)
        #plt.legend()
        if first: 
            avg_errors_critical_renorm_str = "data/rg_paper/sine_wave/avg_errors_critical_renorm_" + str(i+20) + ".dat"
            np.savetxt(avg_errors_critical_renorm_str, avg_errors_critical_renorm, delimiter=",")
            avg_errors_control1_renorm_str = "data/rg_paper/sine_wave/avg_errors_control1_renorm_" + str(i+20) + ".dat"
            np.savetxt(avg_errors_control1_renorm_str, avg_errors_control1_renorm, delimiter=",")
            avg_errors_control2_renorm_str = "data/rg_paper/sine_wave/avg_errors_control2_renorm_" + str(i+20) + ".dat"
            np.savetxt(avg_errors_control2_renorm_str, avg_errors_control2_renorm, delimiter=",")
        else: 
            avg_errors_critical_renorm_str = "data/rg_paper/mg/avg_errors_critical_renorm_" + str(i) + ".dat"
            np.savetxt(avg_errors_critical_renorm_str, avg_errors_critical_renorm, delimiter=",")
            avg_errors_control1_renorm_str = "data/rg_paper/mg/avg_errors_control1_renorm_" + str(i) + ".dat"
            np.savetxt(avg_errors_control1_renorm_str, avg_errors_control1_renorm, delimiter=",")
            avg_errors_control2_renorm_str = "data/rg_paper/mg/avg_errors_control2_renorm_" + str(i) + ".dat"
            np.savetxt(avg_errors_control2_renorm_str, avg_errors_control2_renorm, delimiter=",")            
        #plt.show()

        # train randomly pruned reservoirs and plot performance

        #avg_errors_critical_prune, _ = training_loop(lsm_critical_copy, num_epochs, input_window_size, mg, learning_rate)

        '''
        plt.figure()
        plt.title("Critical vs Non-Critical Performance after Random Pruning")
        plt.plot(avg_errors_critical_prune, color="red", label="Critical Spiking")
        '''

        #avg_errors_control1_prune, _ = training_loop(lsm_control1_copy, num_epochs, input_window_size, mg, learning_rate)

        '''
        plt.plot(avg_errors_control1_prune, color="black", label="Synchronous Spiking")
        '''

        #avg_errors_control2_prune, _ = training_loop(lsm_control2_copy, num_epochs, input_window_size, mg, learning_rate)

        '''
        plt.plot(avg_errors_control2_prune, color="blue", label="Random Spiking")

        plt.xlabel("Training Step")
        plt.ylabel("Average Error")
        plt.title("Average Error During Training")
        plt.legend()
        plt.show()
        '''

        avg_perf_critical += (1-avg_errors_critical[-1])
        avg_perf_control1 += (1-avg_errors_control1[-1])
        avg_perf_control2 += (1-avg_errors_control2[-1])

        avg_perf_critical_renorm += (1-avg_errors_critical_renorm[-1])
        avg_perf_control1_renorm += (1-avg_errors_control1_renorm[-1])
        avg_perf_control2_renorm += (1-avg_errors_control2_renorm[-1])

        #avg_perf_critical_prune += (1-avg_errors_critical_prune[-1])
        #avg_perf_control1_prune += (1-avg_errors_control1_prune[-1])
        #avg_perf_control2_prune += (1-avg_errors_control2_prune[-1])

        avg_size_critical_renorm += (N-n_renorm_critical)
        avg_size_control1_renorm += (N-n_renorm_control1)
        avg_size_control2_renorm += (N-n_renorm_control2)
        with open("data/rg_paper/avg_size_critical_renorm.dat", "a") as file:
            file.write(str(N-n_renorm_critical)+"\n")
        with open("data/rg_paper/avg_size_control1_renorm.dat", "a") as file:
            file.write(str(N-n_renorm_control1)+"\n")
        with open("data/rg_paper/avg_size_control2_renorm.dat", "a") as file:
            file.write(str(N-n_renorm_control2)+"\n")
        
        #avg_size_critical_prune += (N-n_prune_critical)
        #avg_size_control1_prune += (N-n_prune_control1)
        #avg_size_control2_prune += (N-n_prune_control2)
        if i == (trials-1):
            first = False
            break

    avg_perf_critical /= trials 
    avg_perf_control1 /= trials 
    avg_perf_control2 /= trials 

    avg_perf_critical_renorm /= trials 
    avg_perf_control1_renorm /= trials 
    avg_perf_control2_renorm /= trials 

    #avg_perf_critical_prune /= trials 
    #avg_perf_control1_prune /= trials 
    #avg_perf_control2_prune /= trials 

    avg_size_critical_renorm /= trials 
    avg_size_control1_renorm /= trials 
    avg_size_control2_renorm /= trials 

    #avg_size_critical_prune /= trials 
    #avg_size_control1_prune /= trials 
    #avg_size_control2_prune /= trials 


    #plt.figure()

    #plt.plot(N, avg_perf_critical, 'rv') #, label="Critical Start")
    #plt.plot(N, avg_perf_control1, 'bv') #,, label="Sync Start")
    #plt.plot(N, avg_perf_control2, 'kv') #,, label="Random Start")

    #plt.plot(avg_size_critical_renorm, avg_perf_critical_renorm, 'ro') #,, label="Critical End Renorm")
    #plt.plot(avg_size_control1_renorm, avg_perf_control1_renorm, 'bo') #,, label="Sync End Renorm")
    #plt.plot(avg_size_control2_renorm, avg_perf_control2_renorm, 'ko') #,, label="Random End Renorm")

    #plt.plot(avg_size_critical_prune, avg_perf_critical_prune, 'rs') #,, label="Critical End Prune")
    #plt.plot(avg_size_control1_prune, avg_perf_control1_prune, 'ks') #,, label="Sync End Prune")
    #plt.plot(avg_size_control2_prune, avg_perf_control2_prune, 'bs') #,, label="Random End Prune")


    #plt.xlabel("Size of Reservoir", fontsize=18)
    #plt.ylabel("Accuracy After Training", fontsize=18)

print("complete")
#plt.show()
