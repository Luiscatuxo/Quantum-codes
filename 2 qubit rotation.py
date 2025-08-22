### 2 qubit rotation ###

# In this code we simulate and plot the probability of the spin flip of qubit 2 when a specific magnetic field
# design to splin flip qubit 1 is applied. We perform the simulation for different gyromagnetic factors of the 
# second qubit and for different paraemeters embbeded in the magnetic field shape.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Physical constants
c = 2                             # 1 qubit gyromagnetic factor
gamma = np.linspace(0, 20, 100)   # 2 qubit gyromagnetic factor
B0 = 1.0                          # Magnetic field amplitude
k = 0.5                           # Kappa
e = np.linspace(0, 14, 100)       # Eta

# Matrix to store probabilities
P = np.zeros((len(gamma), len(e)))

def psi(u, kappa, eta):
    return kappa*(u+(eta+1)*u**2-2*eta*u**3+eta*u**4)

for i in range(len(gamma)):
    for j in range(len(e)):


        # We define the magnetic field which is the reversed engineered one
        def B_field(t):
            return np.array([-(np.pi/c)*np.sin(psi(t, k, e[j]))-((k*(1+2*(e[j]+1)*t-2*3*e[j]*t**2+4*e[j]*t**3))/2)*np.tan(np.pi*t)*np.cos(psi(t, k, e[j]))+np.sin(np.pi*t)*np.cos(psi(t, k, e[j])), (np.pi/c)*np.cos(psi(t, k, e[j]))-((k*(1+2*(e[j]+1)*t-2*3*e[j]*t**2+4*e[j]*t**3))/2)*np.tan(np.pi*t)*np.sin(psi(t, k, e[j]))+np.sin(np.pi*t)*np.sin(psi(t, k, e[j])), np.cos(np.pi*t)])

        # Bloch vector evolution equation
        def bloch_rhs(t, r):
            B = B_field(t)
            return -1 * gamma[i] * np.cross(r, B)

        # Initial condition: |+⟩ state
        r0 = np.array([1.0, 0.0, 0.0])

        # Simulation time
        s_span = (0, 1)  
        s_eval = np.linspace(s_span[0], s_span[1], 1000)

        # Numerical resolution
        sol = solve_ivp(fun=bloch_rhs, t_span=s_span, y0=r0, t_eval=s_eval)

        # Extract solution
        x, y, z = sol.y
        s = sol.t

        # Compute the probability of being in |-⟩ state
        p = (1+z[-1])/2
        P[i, j] = p

# Plot the propaility using a colormap
plt.figure(figsize=(8, 6))
plt.imshow(P.T, extent=[gamma[0], gamma[-1], e[0], e[-1]], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Probabilidad de spin flip perfecto')
plt.xlabel('gamma_2/gamma_1')
plt.ylabel('eta')
plt.tight_layout()
plt.show()
