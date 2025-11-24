### 1 qubit rotation ###

# In this code we simulate and plot the spin flip of a qubit when a specific reversed engineered magnetic field
# design to make a perfect splin flip is applied. We plot both the three components of the magnetic field as well
# as the three components of the Bloch vector during the state rotation.

import matplotlib.pyplot as plt
import numpy as np

# We define the constants
B_0=1    # Magnetic field amplitude
c = 2    # Gyromagnetic factor

# We define the time scale
s = np.linspace(0,1,10)

# We define the magnetic field components
def B_x(t):
    return -(np.pi/2)*np.sin(t-t**2)-((1-2*t)/2)*np.tan(np.pi*t)*np.cos(t-t**2)+np.sin(np.pi*t)*np.cos(t-t**2)

def B_y(t):
    return (np.pi/2)*np.cos(t-t**2)-((1-2*t)/2)*np.tan(np.pi*t)*np.sin(t-t**2)+np.sin(np.pi*t)*np.sin(t-t**2)

def B_z(t):
    return np.cos(np.pi*t)

def b_x(t):
    return np.sin(np.pi*t)*np.cos(t-t**2)

def b_y(t):
    return np.sin(np.pi*t)*np.sin(t-t**2)

def b_z(t):
    return np.cos(np.pi*t)

# We plot the magnetic field components
plt.figure()
plt.plot(s, B_x(s), label='B_x/B_0')
plt.plot(s, B_y(s), label='B_y/B_0')
plt.plot(s, B_z(s), label='B_z/B_0')
plt.xlabel('s')
plt.legend()

# We plot the Bloch vector components
plt.figure()
plt.plot(s, b_x(s), label='b_x')
plt.plot(s, b_y(s), label='b_y')
plt.plot(s, b_z(s), label='b_z')
plt.xlabel('s')
plt.legend()
plt.show()

