# -*- coding: utf-8 -*-
"""
Wave function and Schrodinger equation
Luis Fernández-Catuxo Ortiz
uo283944
"""

# In this code we simulate the evolution of a particle's wave function in the presence of a harmonic potential
# using the Schrodinger equation. We also compute the evolution of the norm of the wave function,
# position and momentum expected values in time. We also verify that the uncertainty principle is never violated.

# Import the mecessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define all system parameters
xmin=-5                  # Minimum value of the mesh
xmax=5                   # Maximum value of the mesh
ptos=1001                # Number of mesh points
dx=(xmax-xmin)/(ptos-1)  # Mesh step
dt=dx**2/2               # Temporal step 
m=1                      # Particle mass
w=4                      # Angular frecuency

# Define the system's parameters
npasos=50000             # Number of steps 
npasosrep=50             # Number of steps between representations
tpause=0.01              # Pause time between representations

# Creamos the mesh
mallado=np.linspace(xmin,xmax,ptos)

# Define the potential of the harmonic oscillator
V=(1/2)*(w**2)*(mallado**2)       

# Create the wave function to propagate
x0=0       # Where the pulse appears
sigma=0.5  # Gaussians width
k0=10      # Wave number
#fr0=np.exp(-0.5*((mallado-x0)/sigma)**2)*np.cos(k0*mallado) #Parte real de la onda
#fi0=np.exp(-0.5*((mallado-x0)/sigma)**2)*np.sin(k0*mallado) #Parte imaginaria de la onda



############################ FUNCTIONS DEFINITION ##########################
# Define a function to compute the norm
def ctenorm(fr,fi):
    ctenorm=sum((fr**2+fi**2)*dx)
    return ctenorm

# Define a function which normalizes functions
def norm(fr,fi):
    frn=fr/np.sqrt(ctenorm(fr,fi))
    fin=fi/np.sqrt(ctenorm(fr,fi))
    return frn,fin

# Define a function with computes the first derivative
def diff(f):
    diff=np.zeros(ptos)
    diff[0]=(f[1]-f[-1])/(2*dx)
    diff[-1]=(f[0]-f[-2])/(2*dx)
    diff[1:-1]=(f[2:]-f[:-2])/(2*dx)
    return diff

# Define a function with computes the second derivative
def diff2(f):
    diff2=np.zeros(ptos)                       
    diff2[0]=(f[1]+f[-1]-2*f[0])/(dx**2)               
    diff2[-1]=(f[0]+f[-2]-2*f[-1])/(dx**2)           
    diff2[1:-1]=(f[2:]+f[:-2]-2*f[1:-1])/(dx**2) 
    return diff2

# Define a function which computes the mean value of another function
def valormedio(fr,fi,f):
    valormedio=sum((fr**2+fi**2)*f*dx)
    return valormedio
    
# Define a function to compute the mean value of the momentum
def momentum(fr,fi):
    momentum=sum((fr*diff(fi)-fi*diff(fr))*dx)
    return momentum

# Define a function that computes the mean value of the momentum squared
def momentum2(fr,fi):
    momentum2=-sum((fr*diff2(fr)+fi*diff2(fi))*dx)
    return momentum2

# Define a function that computes the Hermite polynomials
def Hermite(x,n):
    if n==0:
        H=1
        return H
    if n==1:
        H=2*x
        return H
    else:
        H=2*x*Hermite(x,n-1)-2*(n-1)*Hermite(x,n-2)
        return H

# Define the Hermite plynomial that we want to use
n=9

# Create the wave function we want to propagate
fr0=np.exp(-0.5*w*(mallado**2))*Hermite(np.sqrt(w)*mallado,n)
fi0=np.zeros(ptos)    

# Print the norm before normalizing
print('Norm before normalization:'+ str(ctenorm(fr0,fi0)))

# Compute the normalized wave functions
fr,fi=norm(fr0,fi0)

# Print thee norm after normalizing
print('Norm after normalization:'+ str(ctenorm(fr,fi)))


# Initialize the time and create lists for the time and norms
t=0
T=[0]
Norma=[ctenorm(fr,fi)]
Valorx=[valormedio(fr,fi,mallado)]
Valorp=[momentum(fr,fi)]
Incertx=[np.sqrt(valormedio(fr,fi,mallado**2)-valormedio(fr,fi,mallado)**2)]
Incertp=[np.sqrt(momentum2(fr,fi)-momentum(fr,fi)**2)]
Incertxp=[np.sqrt(valormedio(fr,fi,mallado**2)-valormedio(fr,fi,mallado)**2)*np.sqrt(momentum2(fr,fi)-momentum(fr,fi)**2)]


############################## CREATE FIGURES ############################
# Create the main figure
figura=plt.figure()
ejes1=figura.add_subplot(211)
funcionr,=ejes1.plot(mallado,fr,'b',label='$\psi_{Re}$(x)')
funcioni,=ejes1.plot(mallado,fi,'r',label='$\psi_{Im}$(x)')
funcionabs,=ejes1.plot(mallado,np.sqrt(fr**2+fi**2),'black',label='$|\psi(x)|$')
funcionpot,=ejes1.plot(mallado,V/max(V),'purple',ls='-.')
plt.xlabel('Position')
plt.ylabel('Amplitude')
ejes1.legend()

# Create another figure to plot the norm
ejes2=figura.add_subplot(234)
normadefuncion,=ejes2.plot(T,Norma,'black',label='Norm')
plt.xlabel('Time')
ejes2.legend()

# Create another figure to plot the mean values
ejes3=figura.add_subplot(235)
valoresperadox,=ejes3.plot(T,Valorx,'r',label='<x>')
valoresperadop,=ejes3.plot(T,Valorp,'b',label='<p>')
plt.xlabel('Time')
ejes3.legend()

# Create another figure to plot the mean
ejes4=figura.add_subplot(236)
indeterminacionx,=ejes4.plot(T,Incertx,'r',label='$\Delta$x')
indeterminacionp,=ejes4.plot(T,Incertp,'b',label='$\Delta$p')
indeterminacionxp,=ejes4.plot(T,Incertxp,'black',label='$\Delta$x·$\Delta$p')
plt.xlabel('Time')
ejes4.legend()



################################# SIMULATION ##################################
# Simulate npasos times
for n in range(npasos):
    
    # Propagation equations
    fi=fi+(dt/4)*diff2(fr)-(dt/2)*V*fr
    
    fr=fr-(dt/2)*diff2(fi)+dt*V*fi
    
    fi=fi+(dt/4)*diff2(fr)-(dt/2)*V*fr
    
    # Update the time
    t=t+dt
    
    # We represent every npasosrepre iterations
    if n%npasosrep==0:
        
        # Compute the uncertainties for each time
        incertx=np.sqrt(valormedio(fr,fi,mallado**2)-valormedio(fr,fi,mallado)**2)
        incertp=np.sqrt(momentum2(fr,fi)-momentum(fr,fi)**2)
        incertxp=incertx*incertp
        
        # Insert the necessary values in the lists for representing
        T.append(t)
        Norma.append(ctenorm(fr,fi))
        Valorx.append(valormedio(fr,fi,mallado))
        Valorp.append(momentum(fr,fi))
        Incertx.append(incertx)
        Incertp.append(incertp)
        Incertxp.append(incertxp)
        
        # Update the real and imaginary part of the wave
        funcionr.set_data(mallado,fr)
        funcioni.set_data(mallado,fi)
        funcionabs.set_data(mallado,np.sqrt(fr**2+fi**2))
        ejes1.set_xlim(xmin,xmax)
        
        # Update the norm
        normadefuncion.set_data(T,Norma)
        ejes2.set_ylim(min(Norma),max(Norma))
        ejes2.set_xlim(0,T[-1])
        
        # Update the mean values
        valoresperadox.set_data(T,Valorx)
        valoresperadop.set_data(T,Valorp)
        ejes3.set_ylim(min(Valorp+Valorx),max(Valorp+Valorx))
        ejes3.set_xlim(0,T[-1])
        
        # Update the uncertainty values
        indeterminacionx.set_data(T,Incertx)
        indeterminacionp.set_data(T,Incertp)
        indeterminacionxp.set_data(T,Incertxp)
        ejes4.set_xlim(0,T[-1])
       
        plt.pause(tpause)
    