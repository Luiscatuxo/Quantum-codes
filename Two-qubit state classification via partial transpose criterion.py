# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:17:50 2025

@author: DELL LATITUDE 7480
"""

# This code studies two–qubit quantum states defined by a base density matrix plus deformations in parameters
# x and y, for each point on a grid it builds rho matrix and checks if it is a valid physical state (positive eigenvalues),
# then applies the partial transpose criterion: Negative eigenvalues → non-physical. 
# Positive but negative under partial transpose → entangled. Otherwise → separable. Finally, it visualizes
# the classification on a color map and prints eigenvalues/eigenvectors of a test matrix.

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as lin 
from matplotlib.colors import ListedColormap, BoundaryNorm 

# Define the probability and the meshgrid 
p=0.75
x=np.linspace(-0.35,0.35,70)
y=np.linspace(-0.35,0.35,70)
X, Y = np.meshgrid(x, y)

# Define Puli matrices
Sigmax=np.array([[0, 1], 
                   [1, 0]])

Sigmay=np.array([[0, -1.j], 
                   [1.j, 0]])

Sigmaz=np.array([[1, 0], 
                   [0, -1]])

# Define matrices M1 and M2
M1=np.kron(Sigmax,Sigmax)
M2=np.kron(Sigmay,Sigmay)

phiphi=np.array([[0.5,0,0,0.5], 
                   [0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]])

Id=np.array([[1,0,0,0], 
                   [0,1,0,0],[0,0,1,0],[0,0,0,1]])

prueba=np.array([[0,0,0,0], 
                   [0,2,-2,0],[0,-2,2,0],[0,0,0,0]])


# Define central density matrix
rho0=(1-p)*phiphi+(p/4)*Id

# Function that computes rho for each (x, y)
def Rho(x, y, M1, M2):
    return rho0 + x * M1 + y * M2

# Matrix to store the different types of state
result_matrix = np.zeros_like(X)

# Function which partially transposes on the second qubit
def partial_transpose(rho):

    # Reorganize indexes
    rho_pt = np.zeros_like(rho, dtype=complex)
    rho_pt[0, 0] = rho[0, 0]
    rho_pt[0, 1] = rho[1, 0]
    rho_pt[0, 2] = rho[0, 2]
    rho_pt[0, 3] = rho[1, 2]
    
    rho_pt[1, 0] = rho[0, 1]
    rho_pt[1, 1] = rho[1, 1]
    rho_pt[1, 2] = rho[0, 3]
    rho_pt[1, 3] = rho[1, 3]
    
    rho_pt[2, 0] = rho[2, 0]
    rho_pt[2, 1] = rho[3, 0]
    rho_pt[2, 2] = rho[2, 2]
    rho_pt[2, 3] = rho[3, 2]
    
    rho_pt[3, 0] = rho[2, 1]
    rho_pt[3, 1] = rho[3, 1]
    rho_pt[3, 2] = rho[2, 3]
    rho_pt[3, 3] = rho[3, 3]
    
    return rho_pt


# Fill the matrix with the pricnipal eigenvector of rho for each (x, y)
for i in range(len(x)):
    for j in range(len(y)):
        
        # Compute rho for each x and y values
        rho = Rho(X[i, j], Y[i, j], M1, M2)
        
        # Compute rho eigenvalues
        autovalores = np.linalg.eig(rho)[0]
        
        # Verify if there is a negative eigenvalue
        if np.any(np.real(autovalores) < 0):  # Verify the real part
            result_matrix[i, j] = -1  # Not a physical state
            
        else:
            # Partial transpose on the second qubit
            rho_pt = partial_transpose(rho)
            
            # CCompute the eigenvalues of the partial transpose
            autovalorespt = np.linalg.eig(rho_pt)[0]
            
            # Verify if there is a negative eigenvalue
            if np.any(autovalorespt < 0):  
                result_matrix[i, j] = 1  # Entangled state
            
            else:
                result_matrix[i, j] = 0   # Separable state
        
plt.imshow(result_matrix, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', aspect='auto')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')

# Invert the y axis
plt.gca().invert_yaxis()

# Add colormap
cmap = plt.get_cmap('viridis')
colores = {
    -1: cmap(0.0),  # Color for -1 
    0: cmap(0.5),  # Color for 0 
    1: cmap(1.0)   # Color for 1
}

for value, color in colores.items():
    plt.scatter([], [], color=color, label={-1: 'Non physical', 0: 'Physical and separable', 1: 'Physical and entangled'}[value])

plt.legend()

# Plot the graph
plt.show()


def calcular_autovalores_autovectores(matriz):
    # Compute the eigenvalues and eigenvectors
    autovalores, autovectores = np.linalg.eig(matriz)
    return autovalores, autovectores


print("Eigenvalues:")
print(calcular_autovalores_autovectores(prueba)[0])
print("\nEigenvectors:")
print(calcular_autovalores_autovectores(prueba)[1])