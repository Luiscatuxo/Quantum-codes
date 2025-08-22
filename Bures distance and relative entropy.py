# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:37:52 2025

@author: DELL LATITUDE 7480
"""
# In this code we compute and plot both the Bures distance and the relative entropy of two density matrix after
# evolving one with a certain hamiltonian.

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, logm

# Define the time mesh
t=np.linspace(0,10,1000)

#Define the A matrix
A=np.array([[0.5, -0.3], 
                   [-0.3, 0.5]])

#Define the Sigma matrix
Sigma=np.array([[1, 0], 
                   [0, -1]])

# Define the time volution operator and its complex conjugate
U=np.array([expm(-1j*Sigma*t_i) for t_i in t])
Udagger=np.array([expm(1j*Sigma*t_i) for t_i in t])


B = np.array([U_i @ A @ Udagger_i for U_i, Udagger_i in zip(U, Udagger)])

# Compute the Bures distance
F = np.array([(np.trace(sqrtm(sqrtm(A) @ B_i @ sqrtm(A))))**2 for B_i in B])
D = np.sqrt(2*(1-np.sqrt(F)))

# Compute the relative entropy
S=np.array([-np.trace(A@logm(B_i))+np.trace(A@logm(A)) for B_i in B])

# Plot the Bures distance and the relative entropy
plt.figure()
plt.title('Bures distance and relative entropy')
plt.plot(t,D, label='$D_{B}(\\rho_{1}(t),\\rho_{10})$', linewidth=2)
plt.plot(t,S, label='$S(\\rho_{1}(t)||\\rho_{10})$', linewidth=2)
plt.xlabel('t',fontsize='12')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


