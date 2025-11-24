### 2 qubit rotation Without interaction###

# In this code we simulate the trajectorty of a second qubit when a specific magnetic field
# design to splin flip qubit 1 is applied on it.

# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from qutip import Bloch

# Define physical constants
g1 = 2       # 1st qubit gyromagnetic factor
tf = 1       # Final time to complete the spin flip
gamma = 5.34 # 2nd qubit gyromagnetic factor
B0 = 1.0     # Z-component magnetic field amplitude
k = 0.5      # Kappa parameter
e = 5        # Eta parameter

# Define the psi function
def psi(u, kappa, eta):
    return kappa*(u + (eta - 1)*u**2 - 2*eta*u**3 + eta*u**4)

# Define the magnetic field which is the reversed engineered one for the 1st qubit
def B_x(t):
    return -(np.pi/(g1*tf))*np.sin(psi(t, k, e))-((k*(1+2*(e-1)*t-6*e*t**2+4*e*t**3))/(g1*tf))*np.tan(np.pi*t)*np.cos(psi(t, k, e)) + B0*np.sin(np.pi*t)*np.cos(psi(t, k, e)) 
def B_y(t):
    return (np.pi/(g1*tf))*np.cos(psi(t, k, e))-((k*(1+2*(e-1)*t-6*e*t**2+4*e*t**3))/(g1*tf))*np.tan(np.pi*t)*np.sin(psi(t, k, e)) + B0*np.sin(np.pi*t)*np.sin(psi(t, k, e))
def B_z(t):
    return B0*np.cos(np.pi*t)

# Define the Schrodinger equation system
def system(t, y):
    c1, c2 = y
    H = (gamma/2) * np.array([[B_z(t), B_x(t)-1j*B_y(t)],
                  [B_x(t)+1j*B_y(t), -B_z(t)]], dtype=complex)
    dydt = -1j * H.dot(np.array([c1, c2]))
    return dydt

# Set the initial conditions: state |↑>
y0 = np.array([1+0j, 0+0j])

# Define the time interval and the evaluation span
t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the system with solve_ivp
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Extract the components of the superposition state from the solution
c1, c2 = sol.y

# Compute the bloch vector components
bx = 2 * np.real(np.conj(c1) * c2)
by = 2 * np.imag(np.conj(c1) * c2)
bz = np.abs(c1)**2 - np.abs(c2)**2

# Compute the probability of being in |-⟩ state and print it
p = np.abs(c2[-1])**2
print(p)

# Plot the trajectory on the bloch sphere
b = Bloch(figsize=[4, 4])
b.add_points(np.array([bx, by, bz]))                        # Plot only the surface points
b.point_color=['red']                                       # Select the color of the points
b.point_size = [1]                                          # Select the size of the points
b.sphere_alpha = 0                                          # we make the bloch sphere quite transparent
b.frame_width = 0                                           # Eliminate the frame
b.xlabel = ['', '']
b.ylabel = ['', '']
b.zlabel = [r'$|\uparrow\rangle$', r'$|\downarrow\rangle$']
b.font_size = 30                                            # Select fiont size
b.show()                                                    # Show the Bloch sphere
plt.show(block=True)                                        # Show the results