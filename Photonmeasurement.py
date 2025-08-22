
# This code is a part of a entangled-photon measurement experiment performed in the lab where we plot the number
# of coincidendes against the delay and fit the data to a gaussian. After doing the fit, we compute the correlation
# time of the entangled photons using two definitions.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
counts = [2443, 2475, 2534, 2597, 2677, 14186, 18304, 18805, 18304, 14186, 2677, 2597, 2534, 2475, 2443]
delay = [-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
xfit = np.linspace(-14, 14, 100)

# Define the gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# Curve fit
popt, pcov = curve_fit(gaussian, delay, counts, p0=[max(counts), 0, 4])

# Adjusted parameters
a, b, c = popt

# Parameter errors
errors = np.sqrt(np.diag(pcov))  # Standard deviations of the parameters
a_err, b_err, c_err = errors

# Compute correlation times
time_correlation = b + np.sqrt(2) * c  # Where the curve drops to 1/e of its maximum
time_correlation1 = b + c * np.sqrt(2 * np.log(2))  # Where the curve drops half the maximum

err_time = b_err + np.sqrt(2)*c_err
err_time1 = b_err + np.sqrt(2+np.log(2))*c_err

# Plot the data and the fit
plt.figure()
plt.plot(delay, counts, 'o', markersize=5, label='Data')
plt.plot(xfit, gaussian(xfit, *popt), label=f'Gaussian fit', color = '#1f77b4')

# Mark the correlation times
plt.axvline(x=time_correlation, color='black', linestyle='--', label=f'corr. time 1/e')
plt.axvline(x=time_correlation1, color = 'orange',linestyle='--', label=f'corr. time FWHM')

plt.xlabel('Delay (ns)')
plt.ylabel('Coincidences')
plt.legend()
plt.show()

# Print the correlation times
print(f"Correlation time 1/e: {time_correlation:.4f} +- {err_time:.4f}")
print(f"Correlation time FWHM: {time_correlation1:.4f} +- {err_time1:.4f}")
