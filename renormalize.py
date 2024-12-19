import numpy as np


def renormalize(W, V, w_th):
    N = W.shape[0]  
    merged = True
    while(merged):
        merged = False 
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > w_th:
                    merged = True 
                    print(f"Merging neurons {i} and {j} with weight {W[i, j]}")

                    V[i] = (V[i] + V[j]) / 2

                    for k in range(N):
                        if k != i and k != j:
                            if W[i, k] * W[j, k] == 0:
                                W[i,k] = W[k,i] = max(W[i, k], W[j, k])
                            else:
                                W[i,k] = W[k,i] = (W[i,k] + W[j,k])/2

                    W[j, :] = 0
                    W[:, j] = 0
                    V[j] = 0

    return W, V

np.set_printoptions(linewidth=np.inf)

N = 9
W = np.random.rand(N, N)
W = (W + W.T) / 2  # Symmetrize the weight matrix
np.fill_diagonal(W, 0)  # No self-connections
V = np.random.rand(N)

print("Initial weight matrix:\n", W)
print("Initial neuron states:\n", V)

w_th = 0.7
W_new, V_new = renormalize(W, V, w_th)

print("\nUpdated weight matrix:\n", W_new)
print("Updated neuron states:\n", V_new)

