import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral_frac as sf

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

img = cv.imread("data/bass.jpg", cv.IMREAD_GRAYSCALE)
img=img[:, :len(img)]

_, fig = plt.subplots(2,3)

popt, kvals_fit, Abins_fit = sf.calc_power_spectrum(img)
# Extract the fitted beta value
beta = popt[0]
     
alpha = beta / 2

# Log-log plot of the binned power spectrum
fig[0,0].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,0].set_yticks([])
fig[0,0].set_yticks([], minor=1)
fig[0,0].set_title("PSD")
'''
fig[0,0].set_xlabel("$k$")
fig[0,0].set_ylabel("$P(k)$")
'''
print(f"Estimated slope: {beta}")
print(f"Estimated alpha: {alpha}")

diff_img = sf.fractional_diff_2d(img, alpha)

# Plot the fitted power law
fig[0,0].loglog(kvals_fit, sf.power_law(kvals_fit, *popt), 'r--', label=f'Fit: $\\beta={beta:.2f}$')
#fig[0,0].set_legend()
_, kvals_fit, Abins_fit = sf.calc_power_spectrum(diff_img)
fig[0,1].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
'''
fig[0,1].set_xlabel("$k$")
fig[0,1].set_ylabel("$P(k)$")
'''
fig[0,1].set_yticks([])
fig[0,1].set_title("PSD")


w_img = isolate_whitened_portion(diff_img)


res_img = 2*img+(img - 5*w_img)



_, kvals_fit, Abins_fit = sf.calc_power_spectrum(res_img)
fig[0,2].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
'''
fig[0,2].set_xlabel("$k$")


fig[0,2].set_ylabel("$P(k)$")
'''
fig[0,2].set_yticks([])
fig[0,2].set_title("PSD")



fig[1,0].imshow(img, cmap='gray')
fig[1,0].set_title("Original Image")
fig[1,1].imshow(diff_img, cmap='gray')
fig[1,1].set_title("Whitened Image")
fig[1,2].imshow(res_img, cmap='gray')
fig[1,2].set_title("Residual Image")

plt.show()

