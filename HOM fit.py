
# In this code we fit HOM dip data obtained in the lab to an inverted gaussian function
# and plot both the data and the fit. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Reads the text file
filename = r'C:\Users\DELL LATITUDE 7480\Desktop\linear-scan-data_01.txt'

data = np.loadtxt(filename, skiprows=5)
#D=1e13
#01 = 152
#02 = 148
#03 = todo
#04 = 74

# Use the relevant columns
position = data[:152, 0] # First column (position in mm)

coincidences = data[:152, 5]  # Sixth column (coincidences)
delay = (position/300)+(0.82346/300) # Delay in ns and displaced to have the 0 in the center

# Define the inverted gaussian function
def inverted_gaussian(dt, A, D):
    return A*0.5* (1-np.exp(-(dt*D)**2))

# Adjust the data to the inverted gaussian
popt, pcov = curve_fit(inverted_gaussian, delay, coincidences, p0=[3200, 1e4])

# Optimized parameters for the fit and their errors
A_opt, D_opt = popt
A_err, D_err = np.sqrt(np.diag(pcov))

# Plot the curve
y_fit = inverted_gaussian(delay, A_opt, D_opt)

#print(1e-9/D_opt)
t=D_opt/(10e8)
err_t = D_err/(D_opt)

print(1e-9/D_opt)
print(1e-9*D_err/(D_opt)**2)
print(D_opt*1e-12/(1e-9))

# Create the plot
plt.plot(delay, coincidences, 'b-', label='Data')
plt.plot(delay, y_fit, '-', color='black', label='Gaussian fit')
#plt.axvline(x=0, linestyle='--')
plt.xlabel('Delay (ns)')
plt.xlim(min(delay),max(delay))
plt.ylabel('$Coincidences\;(s^{-1})$')
plt.legend()
plt.show()