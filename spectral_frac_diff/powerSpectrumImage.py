import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift
import scipy.stats as stats
from scipy.optimize import curve_fit
import cv2 as cv

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

img = cv.imread("trees.jpg", cv.IMREAD_GRAYSCALE)
img=img[:, :len(img)]

_, fig = plt.subplots(2,2)

beta, kvals_fit, Abins_fit = calc_power_spectrum(img)
alpha = beta / 2

# Log-log plot of the binned power spectrum
fig[0,0].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,0].set_xlabel("$k$")
fig[0,0].set_ylabel("$P(k)$")

print(f"Estimated slope: {beta}")
print(f"Estimated alpha: {alpha}")

diff_img = fractional_diff_2d(img, alpha)
# Plot the fitted power law
#fig[0,0].loglog(kvals_fit, power_law(kvals_fit, *popt), 'r--', label=f'Fit: $\\beta={beta:.2f}$')
#fig[0,0].set_legend()
_, kvals_fit, Abins_fit = calc_power_spectrum(diff_img)
fig[0,1].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,1].set_xlabel("$k$")
fig[0,1].set_ylabel("$P(k)$")

fig[1,0].imshow(img, cmap='gray')
fig[1,1].imshow(diff_img, cmap='gray')
plt.show()

