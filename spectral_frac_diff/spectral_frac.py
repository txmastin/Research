import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
import scipy.stats as stats
from scipy.optimize import curve_fit

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
    # fourier transform and power spectrum
    fourier_img = np.fft.fftn(img)
    fourier_amplitudes = np.abs(fourier_img)**2

    # create the frequency grid and normalize
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2d = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2d[0]**2 + kfreq2d[1]**2)

    # flatten the arrays
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # bin the power spectrum
    kbins = np.arange(0.5, npix//2+1, 1)
    #kbins = np.logspace(np.log10(0.5), np.log10(npix // 2), 50)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)

    # adjust the amplitude to account for bin area
    abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    fit_range = (kvals > 20) & (kvals < 2800000)
    kvals_fit = kvals[fit_range]
    abins_fit = abins[fit_range]
    # fit the power law using curve_fit
    popt, pcov = curve_fit(power_law, kvals_fit, abins_fit, p0=[2, 10**14])
    return popt, kvals_fit, abins_fit

