import numpy as np
import matplotlib.pyplot as plt

width = 20
height = 20

num_steps = 50

class Node:
    def __init__(self):
        self.weights = [[np.random.rand() for _ in range(width)] for _ in range(height)]
        self.state = -1

class Net:
    def __init__(self, prob):
        self.nodes = [[Node() for _ in range(width)] for _ in range(height)]
        
        # randomly set an initial state based on 'prob'
        for i in range(width):
            for j in range(height):
                if prob > np.random.rand():
                    self.nodes[i][j].state = 1

    def get_states(self):
        states = [[x.state for x in row] for row in self.nodes]
        return states

    def get_weights(self):
        weights = [[x.weights for x in row] for row in self.nodes]
        return weights

    def calculate_h(self, x, y):
        h = 0
        for i in range(width):
            for j in range(height):
                if i == x:
                    if j == y:
                        break
                h += self.nodes[x][y].state*self.nodes[i][j].state*self.nodes[x][y].weights[i][j]
        return h
'''
h = sum(x_i*x_j+w_ij) for i != j
x_i = -1 if h < 0 else +1


'''
x = np.linspace(0,1,50)

act_rec = []
for prob in x:
    
    net = Net(prob)

    for n in range(num_steps):
        for i in range(width):
            for j in range(height):
                if (net.calculate_h(i, j)) < 0:
                    net.nodes[i][j].state = -1
                else:
                    net.nodes[i][j].state = 1
    act = np.sum(net.get_states())   
    act_rec.append(act)

plt.plot(x, act_rec, 'bo')
plt.show()
