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
import torch.nn.functional as F
import spectral_frac as sf

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
    frac_operator = (1j * 2 * np.pi * radius)**alpha
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

    fit_range = (kvals > 10) & (kvals < 280000)
    kvals_fit = kvals[fit_range]
    Abins_fit = Abins[fit_range]
    # Fit the power law using curve_fit
    popt, pcov = curve_fit(power_law, kvals_fit, Abins_fit, p0=[2, 10**14])

    # Extract the fitted beta value
    beta = popt[0]
        
    return beta, kvals_fit, Abins_fit

def isolate_whitened_portion(frac_diff_img, threshold=5):

    # Perform Fourier transform
    F = np.fft.fft2(frac_diff_img)
    F_shifted = np.fft.fftshift(F)  # Shift zero frequency to center

    # Compute power spectral density (PSD)
    PSD = np.abs(F_shifted)**2

    # Identify frequencies with approximately flat PSD (whitening regions)
    mean_psd = np.mean(PSD)
    whitening_mask = (PSD < (1 + threshold) * mean_psd) & (PSD > (1 - threshold) * mean_psd)

    # Apply whitening mask
    F_whitened = F_shifted * whitening_mask

    # Inverse Fourier transform to get spatial domain whitened portion
    F_whitened_shifted = np.fft.ifftshift(F_whitened)
    whitened_img = np.fft.ifft2(F_whitened_shifted).real

    return whitened_img



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
        w_img = isolate_whitened_portion(img_frac)
        res_img = 2*img+(img - 5*w_img)
        img_tensor = torch.tensor(res_img, dtype=torch.float32).unsqueeze(0)
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

# Fully Connected Network (FCN)
class SimpleFCN(nn.Module):
    def __init__(self, input_size=32*32, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Input size depends on flattened image size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here; logits are returned
        return x


# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 124)
        self.fc2 = nn.Linear(124, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


average = 0
diff_arr = []
tests = 20
for i in range(tests):
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
        #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

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
    frac_acc = 100*correct/total
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
        #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

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
    nfrac_acc = 100*correct/total
    print(f"Test Accuracy Without Fractional Differentiation: {100 * correct / total:.2f}%")
    
    diff = frac_acc - nfrac_acc
    print("different: ", diff)
    diff_arr.append(diff)
    average = sum(diff_arr)/(i+1)
    print("average: ", average)
