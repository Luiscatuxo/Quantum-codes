### In this code we simulate the trajectory of the bloch vector of one qubit under the effect
# of a reversed engineered magnetic field in order to perform a perfect spin flip ###

# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from qutip import Bloch
#plt.rcParams['text.usetex'] = True

# Define the constants of our simulation
B_0 = 1 # Z-component magnetic field amplitude
tf = 1  # Final time to complete the spin flip
g1 = 2  # Gyromagnetic factor of the qubit

# Define the time scale
s = np.linspace(0, 1, 500)

# Define the reversed engineered magnetic field components
def B_x(t):
    return 0*t
#-(np.pi/(g1*tf))*np.sin(t-t**2) - ((1-2*t)/g1)*np.tan(np.pi*t)*np.cos(t-t**2) + B_0*np.sin(np.pi*t)*np.cos(t-t**2)
def B_y(t): 
    return np.pi/(g1*tf)+0*t
#(np.pi/(g1*tf))*np.cos(t-t**2) - ((1-2*t)/g1)*np.tan(np.pi*t)*np.sin(t-t**2) + B_0*np.sin(np.pi*t)*np.sin(t-t**2)
def B_z(t):
    return 0*t
#B_0*np.cos(np.pi*t)

# Define the Schrodinger equation system
def system(t, y):
    c1, c2 = y

    # Define the hamiltonian
    H = (g1/2) * np.array([[B_z(t), B_x(t)-1j*B_y(t)],
                  [B_x(t)+1j*B_y(t), -B_z(t)]], dtype=complex)
    
    # Write the differential equation
    dydt = -1j * H.dot(np.array([c1, c2]))
    return dydt

# Set the initial conditions: state |↑>
y0 = np.array([1+0j, 0+0j])

# Define the time interval and the evaluation span
t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the system with solve_ivp
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Extract the components of the superposition state from the solution
c1, c2 = sol.y

# Compute the bloch vector components
bx = 2 * np.real(np.conj(c1) * c2)
by = 2 * np.imag(np.conj(c1) * c2)
bz = np.abs(c1)**2 - np.abs(c2)**2

# Plot the trajectory on the bloch sphere
b = Bloch(figsize=[4, 4])
b.add_points(np.array([bx, by, bz]))                        # Plot only the surface points
b.point_color=['r']                                         # Select the color of the points
b.point_size = [1]                                          # Select the size of the points
b.sphere_alpha = 0                                          # we make the bloch sphere quite transparent
b.frame_width = 0                                           # Eliminate the frame
b.xlabel = ['', '']
b.ylabel = ['', '']
b.zlabel = [r'$|\uparrow\rangle$', r'$|\downarrow\rangle$']
b.font_size = 30                                            # Select fiont size
b.show()                                                    # Show the Bloch sphere
plt.show(block=True)                                        # Show the results