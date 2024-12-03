import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral_frac as sf


img = cv.imread("data/bass.jpg") 
img=img[:, :len(img)]
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

r, g, b = cv.split(img)

_, fig = plt.subplots(2,2)

popt, kvals_fit, Abins_fit = sf.calc_power_spectrum(b)
# Extract the fitted beta value
beta = popt[0]
     
alpha = beta / 2
b = sf.fractional_diff_2d(b, alpha)
g = sf.fractional_diff_2d(g, alpha)
r = sf.fractional_diff_2d(r, alpha)
'''
b = b / 10
g = g / 10
r = r / 10
'''
# Log-log plot of the binned power spectrum
fig[0,0].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,0].set_xlabel("$k$")
fig[0,0].set_ylabel("$P(k)$")

print(f"Estimated slope: {beta}")
print(f"Estimated alpha: {alpha}")

diff_img = cv.merge([r, g, b])

# Plot the fitted power law
fig[0,0].loglog(kvals_fit, sf.power_law(kvals_fit, *popt), 'r--', label=f'Fit: $\\beta={beta:.2f}$')
#fig[0,0].set_legend()
_, kvals_fit, Abins_fit = sf.calc_power_spectrum(b)
fig[0,1].loglog(kvals_fit, Abins_fit, label='Binned Power Spectrum')
fig[0,1].set_xlabel("$k$")
fig[0,1].set_ylabel("$P(k)$")

fig[1,0].imshow(img)
fig[1,1].imshow((diff_img).astype(np.uint8))
plt.show()

