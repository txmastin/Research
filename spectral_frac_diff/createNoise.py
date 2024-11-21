import numpy as np
import matplotlib.pyplot as plt


# Parameters for the power spectrum
size = 2000  # Size of the 2D grid (size x size)
alpha = 3.15  # Exponent of the power law

# Generate a grid of frequencies
kx = np.fft.fftfreq(size).reshape(-1, 1)
ky = np.fft.fftfreq(size).reshape(1, -1)
k = np.sqrt(kx**2 + ky**2)
k[0, 0] = 1  # Avoid division by zero

# Generate power spectrum following a power law
power_spectrum = k ** -alpha

# Generate random phases
random_phases = np.exp(2j * np.pi * np.random.rand(size, size))

# Combine magnitude and phase to create a complex field
complex_field = np.sqrt(power_spectrum) * random_phases

# Perform inverse FFT to obtain the spatial representation
spatial_field = np.fft.ifft2(complex_field).real

# Normalize the field for visualization
spatial_field_normalized = (spatial_field - np.min(spatial_field)) / (np.max(spatial_field) - np.min(spatial_field))

# Plot the generated image
plt.figure(figsize=(12, 12))
plt.imshow(spatial_field_normalized, cmap='gray', origin='lower')
plt.axis("off")
plt.savefig('data/noise/noise.jpg', bbox_inches='tight', pad_inches=0)
#plt.title("Image with Power-Law Power Spectrum")
#plt.show()

