import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from torchvision import datasets, transforms
import numpy as np
from numpy.polynomial.polynomial import polyfit

# Training Parameters
batch_size = 128
data_path = '/tmp/data/mnist'
num_classes = 10

# Torch Variables
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Temporal dynamics
beta = 0.8
num_steps = 25
threshold = 1.3

# Network architecture
num_inputs = 28 * 28
num_hidden = 1500
num_outputs = 10


# Download MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0), (1))
])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)  # time-steps x batch x num_out

# Load network to device
net = Net().to(device)

# Training network
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))

num_epochs = 1
counter = 0

# For avalanche tracking
avalanche_sizes = []

for epoch in range(num_epochs):
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        net.train()
        spk_rec = net(data)

        # Count spikes to detect avalanches
        for t in range(num_steps):
            spike_count = spk_rec[t].sum().item()
            if spike_count > 0:
                avalanche_sizes.append(spike_count)

        # Loss calculation
        loss_val = loss(spk_rec.sum(0), targets)

        # Gradient calculation and weight updates
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if counter % 10 == 0:
            print(f"Iteration: {counter} \t Train loss: {loss_val.item()}")
        counter += 1

# Testing
total = 0
correct = 0

with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)

    # forward pass
    test_spk = net(data.view(data.size(0), -1))

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

uniq, count = np.unique(avalanche_sizes, return_counts=True)
print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

# Plot avalanche sizes
#plt.hist(avalanche_sizes, bins=50, color='blue', alpha=0.7)

plt.scatter(uniq, count, color='blue', alpha=0.7)
plt.title('Neuronal Avalanche Size Distribution')
plt.xlabel('Avalanche Size')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.show()

