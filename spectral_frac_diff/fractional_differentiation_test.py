import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift
import scipy.stats as stats
from scipy.optimize import curve_fit
import cv2 as cv
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

def power_law(k, alpha, c):
    return c * k**(-alpha)

# Function to apply fractional differentiation in the frequency domain
def fractional_diff_2d(image, alpha):
    N, M = image.shape
    u = np.fft.fftfreq(N)  # Frequency coordinates for rows
    v = np.fft.fftfreq(M)  # Frequency coordinates for columns
    U, V = np.meshgrid(u, v, indexing="ij")
    radius = np.sqrt(U**2 + V**2)  # Distance from origin in frequency space
    radius[0, 0] = 1  # Avoid division by zero at DC component

    # Apply fractional differentiation operator in frequency domain
    fft_image = fft2(image)
    frac_operator = (2 * np.pi * radius)**alpha
    diff_image = ifft2(fft_image * frac_operator).real
    return diff_image

def calc_power_spectrum(img):
    npix = np.size(img[0])
    # Fourier transform and power spectrum
    fourier_img = np.fft.fftn(img)
    fourier_amplitudes = np.abs(fourier_img)**2

    # Create the frequency grid and normalize
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    # Flatten the arrays
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # Bin the power spectrum
    kbins = np.arange(0.5, npix//2+1, 1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)

    # Adjust the amplitude to account for bin area
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    fit_range = (kvals > 0) & (kvals < 18000)
    kvals_fit = kvals[fit_range]
    Abins_fit = Abins[fit_range]
    # Fit the power law using curve_fit
    popt, pcov = curve_fit(power_law, kvals_fit, Abins_fit, p0=[1, 10**12])

    # Extract the fitted beta value
    beta = popt[0]
        
    return beta, kvals_fit, Abins_fit

def preprocess_data(dataset, fractional_diff_2d):
    preprocessed_data = []
    labels = []
    i = 0
    for img, label in dataset:
        print(i)
        i+=1
        img = img.squeeze(0).numpy()
        img_np = np.array(img)
        beta,_,_ = calc_power_spectrum(img_np)
        alpha = beta / 2
        img_frac = fractional_diff_2d(img_np, alpha)
        img_tensor = torch.tensor(img_frac, dtype=torch.float32).unsqueeze(0)
        preprocessed_data.append(img_tensor)
        labels.append(label)
    
    return torch.stack(preprocessed_data), torch.tensor(labels)


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()  # Convert the image to a tensor
    #transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std of 0.5 for each channel
])

# Load the training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Load the test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



# Prepare datasets
frac_train_data, frac_train_labels = preprocess_data(train_dataset, fractional_diff_2d)
frac_test_data, frac_test_labels = preprocess_data(test_dataset, fractional_diff_2d)

frac_train_loader = DataLoader(TensorDataset(frac_train_data, frac_train_labels), batch_size=32, shuffle=True)
frac_test_loader = DataLoader(TensorDataset(frac_test_data, frac_test_labels), batch_size=32, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)  # Assuming 10 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in frac_train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in frac_test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy After Fractional Differentiation: {100 * correct / total:.2f}%")

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy Without Fractional Differentiation: {100 * correct / total:.2f}%")

