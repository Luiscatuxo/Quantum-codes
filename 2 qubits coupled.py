### 2 qubits coupled ###

# In this code we simulate and plot the trajectory of the bloch vector of qubit 1 when a specific magnetic field
# design to splin flip qubit 1 and crosstalk from qubit 2 is applied.

# Import the necessary libraries 
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from qutip import Bloch

# We define the constants of our simulation
B_0 = 1  # Z-component magnetic field amplitude 
tf = 1   # Final time to complete the spin flip
g1 = 2   # Gyromagnetic factor of qubit 1
g2 = 2   # Gyromagnetic factor of qubit 2
E0 = 1000 # Energy of a high qubit 1 state
J = 0.5  # Couple strengh

# Create some functions to simplify the calculus
gmas = g1 + g2
gmenos = g1 - g2

# Define the time scale
s = np.linspace(0, 1, 500)

# Define the magnetic field components designed to spin flip the qubit 1
def Bx(t):
    return 0*t
#-(np.pi/(g1*tf))*np.sin(t-t**2) - ((1-2*t)/g1)*np.tan(np.pi*t)*np.cos(t-t**2) + B_0*np.sin(np.pi*t)*np.cos(t-t**2)
def By(t):
    return (np.pi/(g1*tf))+0*t
#(np.pi/(g1*tf))*np.cos(t-t**2) - ((1-2*t)/g1)*np.tan(np.pi*t)*np.sin(t-t**2) + B_0*np.sin(np.pi*t)*np.sin(t-t**2)
def Bz(t):
    return 0*t
#B_0*np.cos(np.pi*t)

# Define some useful magnetic fields to simplify calculus
def Bmas(t):
    return Bx(t) + 1j*By(t)
def Bmenos(t):
    return Bx(t) - 1j*By(t)

# Compute the determinant resultant of the Lowdin partition
def det(t):
    return E0**2 - 2*g1*E0*Bz(t) + gmas*gmenos*Bz(t)**2 - 2*g2*Bz(t)*J

# Define the effective hamiltonian components result of the partition
def h00(t):
    return (gmas*Bz(t) + J) - (1/det(t))*(E0 - gmenos*Bz(t) - J)*g1**2*Bmas(t)*Bmenos(t)
def comp1(t):
    return gmas*Bz(t) + J
def comp2(t):
    return - (1/det(t))*(E0 - gmenos*Bz(t) - J)*g1**2*Bmas(t)*Bmenos(t)
def h01(t):
    return (g2*Bmenos(t)) - (1/det(t))*(2*J*E0 - gmenos*Bz(t)*2*J - g1*g2*Bmas(t)*Bmenos(t))*Bmenos(t)*g1
def h10(t):
    return h01(t).conjugate()
def h11(t):
    return (gmenos*Bz(t) - J) - (1/det(t))*(4*J*g2 + E0*g1 - g1*gmas*Bz(t) + J*g1)*g1*Bmas(t)*Bmenos(t) 

# Define the Schrodinger equation system
def system(t, y):
    c1, c2 = y

    # Define the effective hamiltonian
    H = (1/2) * np.array([[h00(t), h01(t)],
                  [h10(t), h11(t)]], dtype=complex)
    
    # Define the differential equation
    dydt = -1j * H.dot(np.array([c1, c2]))
    return dydt

# Set the initial conditions: state |↑>
y0 = np.array([1.0+0j, 0.0+0j])

# Define the time interval and the evaluation span
t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the system with solve_ivp
sol = solve_ivp(system, t_span, y0, t_eval=t_eval, rtol=1e-10, atol=1e-12)

# Extract the components of the superposition state from the solution
c1, c2 = sol.y

# Compute the bloch vector components
bx = 2 * np.real(np.conj(c1) * c2)
by = 2 * np.imag(np.conj(c1) * c2)
bz = np.abs(c1)**2 - np.abs(c2)**2

# Compute the angles theta and phi on the Bloch sphere
#theta = np.arccos(bz)          
#phi = np.arctan2(by, bx)       # original range [-π, π]
#phi = np.mod(phi, 2*np.pi)     # changes the range to [0, 2π)

# Compute the probability of measuring the state in |↓> after the time has passed and print it
P = np.conj(c2[-1])*c2[-1]
print(P)

# Plot the results of the Bloch vector after solving the system
fig, ax = plt.subplots()
ax.plot(s, bx, label='$b_x$')
ax.plot(s, by, label='$b_y$')
ax.plot(s, bz, label='$b_z$')
ax.set_xlabel('$s=t/t_f$', fontsize=20)
ax.legend(fontsize=20, loc='best')
ax.tick_params(axis='both', which='major', labelsize=20)


#plt.figure()
#plt.plot(s, theta, label='theta',fontsize=30)
#plt.plot(s, phi, label='phi',fontsize=30)
#plt.xlabel('s', fontsize=30)
#plt.legend()

fig, ax = plt.subplots()
ax.plot(s, h00(s), label='$h_{00}^{h_eff}$')
ax.plot(s, comp1(s), label='$O(J)$')
ax.set_xlabel('$s=t/t_f$', fontsize=20)
ax.legend(fontsize=20, loc='best')
ax.tick_params(axis='both', which='major', labelsize=20)

# Plot the trajectory on the bloch sphere
b = Bloch(figsize=[4, 4])
b.add_points(np.array([bx, by, bz]))                        # Plot only the surface points
b.point_color=['red']                                         # Select the color of the points
b.point_size = [1]                                          # Select the size of the points
b.sphere_alpha = 0                                          # we make the bloch sphere quite transparent
b.frame_width = 0                                           # Eliminate the frame
b.xlabel = ['', '']
b.ylabel = ['', '']
b.zlabel = [r'$|\uparrow\rangle$', r'$|\downarrow\rangle$']
b.font_size = 30                                            # Select fiont size
b.show()                                                    # Show the Bloch sphere
plt.show(block=True)                                        # Show the results
