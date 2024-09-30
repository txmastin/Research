import numpy as np
import random

prob = 0.01

class RandomBooleanNetwork:
    def __init__(self, N, K):
        self.N = N  # Number of nodes
        self.K = K  # Number of inputs per node (aka connectivity)

        # initial states for each node
        
        # for simple 0 or 1 random states use:
        self.states = np.random.choice([0, 1], N)
        
        '''
        # for probabalistic 0, 1 use:
        self.states = [0 for _ in range(N)]
        for i in range(N):
            if np.random.rand() < prob:
                self.states[i] = 1 
        '''
        
        # sample K choices from 0-N for each node in N
        self.connections = [random.sample(range(N), K) for _ in range(N)]

        self.boolean_functions = [self.generate_boolean_function(K) for _ in range(N)]

    def generate_boolean_function(self, K):
        """Generates a random Boolean function for K inputs."""
        # 2^K possible input combinations -> random 0 or 1 for each
        num_combinations = 2 ** K
        
        bool_func = np.random.choice([0,1], num_combinations)
        return bool_func

    def update(self):
        """Updates the network by applying Boolean functions."""
        new_states = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            # Get the states of the K input nodes for this node
            input_states = tuple(self.states[j] for j in self.connections[i])
            
            # Convert input states to index (like binary to decimal conversion)
            input_index = int(''.join(map(str, input_states)), 2)
            # Update the state of node i using its boolean function
            new_states[i] = self.boolean_functions[i][input_index]
        self.states = new_states

    def run(self, steps=10):
        """Runs the network for a certain number of steps."""
        state_rec = []
        for _ in range(steps):
            self.update()
            state_rec.append(sum(self.states))
        return state_rec

N = 1000 
K = 20
rbn = RandomBooleanNetwork(N, K)

state_rec = rbn.run(steps=1000)
print(np.stack(state_rec))

