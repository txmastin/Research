import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral_frac as sf


img = cv.imread("data/bass.jpg", cv.IMREAD_GRAYSCALE)
img=img[:, :len(img)]

_, fig = plt.subplots(2,2)

popt, kvals_fit, Abins_fit = sf.calc_power_spectrum(img)
# Extract the fitted beta value
beta = popt[0]
     
alpha = beta / 2

# Log-log plot of the binned power spectrum
fig[0,0].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,0].set_xlabel("$k$")
fig[0,0].set_ylabel("$P(k)$")

print(f"Estimated slope: {beta}")
print(f"Estimated alpha: {alpha}")

diff_img = sf.fractional_diff_2d(img, alpha)
# Plot the fitted power law
fig[0,0].loglog(kvals_fit, sf.power_law(kvals_fit, *popt), 'r--', label=f'Fit: $\\beta={beta:.2f}$')
#fig[0,0].set_legend()
_, kvals_fit, Abins_fit = sf.calc_power_spectrum(diff_img)
fig[0,1].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,1].set_xlabel("$k$")
fig[0,1].set_ylabel("$P(k)$")

fig[1,0].imshow(img, cmap='gray')
fig[1,1].imshow(diff_img, cmap='gray')
plt.show()

