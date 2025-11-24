### 2 qubit rotation ###

# In this code we simulate and plot the probability of the spin flip of qubit 2 when a specific magnetic field
# design to splin flip qubit 1 is applied. We perform the simulation for different gyromagnetic factors of the 
# second qubit and for different parameters embbeded in the magnetic field shape.

# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Define physical constants
g1 = 2                           # 1st qubit gyromagnetic factor
tf = 1                           # Final time to complete the spin flip
gamma = np.linspace(0, 10, 100)  # Gyromagnetic factor ratio g2/g1
B0 = 1.0                         # Z-component magnetic field amplitude
k = 3.1                          # Kappa parameter
e = np.linspace(0, 14, 100)      # Eta parameter

# Matrix to store the probabilities
P = np.zeros((len(gamma), len(e)))

# Define the psi function
def psi(u, kappa, eta):
    return kappa*(u + (eta - 1)*u**2 - 2*eta*u**3 + eta*u**4)

# Iterate over all posible values of eta and the gyromagnetic ratio
for i in range(len(gamma)):
    for j in range(len(e)):

        # Define the magnetic field which is the reversed engineered one
        def B_x(t):
            return -(np.pi/(g1*tf))*np.sin(psi(t, k, e[j]))-((k*(1+2*(e[j]-1)*t-6*e[j]*t**2+4*e[j]*t**3))/(g1*tf))*np.tan(np.pi*t)*np.cos(psi(t, k, e[j])) + B0*np.sin(np.pi*t)*np.cos(psi(t, k, e[j])) 
        def B_y(t):
            return (np.pi/(g1*tf))*np.cos(psi(t, k, e[j]))-((k*(1+2*(e[j]-1)*t-6*e[j]*t**2+4*e[j]*t**3))/(g1*tf))*np.tan(np.pi*t)*np.sin(psi(t, k, e[j])) + B0*np.sin(np.pi*t)*np.sin(psi(t, k, e[j]))
        def B_z(t):
            return B0*np.cos(np.pi*t)
        
        # Define the Schrodinger equation system
        def system(t, y):
            c1, c2 = y
            H = ((gamma[i]*g1)/2) * np.array([[B_z(t), B_x(t)-1j*B_y(t)],
                        [B_x(t)+1j*B_y(t), -B_z(t)]], dtype=complex)
            dydt = -1j * H.dot(np.array([c1, c2]))
            return dydt

        # Set the initial conditions: state |↑>
        y0 = np.array([1+0j, 0+0j])

        # Define the time interval and the evaluation span
        t_span = (0, 1)
        t_eval = np.linspace(t_span[0], t_span[1], 100)

        # Solve the system with solve_ivp
        sol = solve_ivp(system, t_span, y0, t_eval=[1])

        # Extract the components of the superposition state from the solution
        c1, c2 = sol.y

        # Compute the bloch vector components
        bx = 2 * np.real(np.conj(c1) * c2)
        by = 2 * np.imag(np.conj(c1) * c2)
        bz = np.abs(c1)**2 - np.abs(c2)**2

        # Compute the probability of being in |-⟩ state and store it in the probability matrix
        p = 1-np.abs(c2)**2
        P[i, j] = p

# Plot the probability using a colormap
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
im = plt.imshow(P.T, extent=[gamma[0], gamma[-1], e[0], e[-1]],
                origin='lower', aspect='auto', cmap='jet')

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label('$\Delta$', fontsize=25)   # tamaño de la etiqueta
cbar.ax.tick_params(labelsize=20)         # tamaño de los números del colorbar

# Labels de los ejes
plt.xlabel('$\gamma_2/\gamma_1$', fontsize=25)
plt.ylabel('$\eta$', fontsize=25)

# Números de los ejes
plt.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.show()
