# -*- coding: utf-8 -*-
"""
Potential barrier, superposition in 1-D and colapse of the wafe function.
Luis Fernández-Catuxo Ortiz
uo283944
"""
# In this code we simulate the dynamic evolution of a particle's wave function when facing a potential barrier
# and simulate the measuring of the particle's position, collapsing its wave function. 

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random as rand


# Define the system's parameters
xmin=-5                  # Minimum value of the mesh
xmax=5                   # Maximum value of the mesh
ptos=1001                # Number of mesh points
dx=(xmax-xmin)/(ptos-1)  # Mesh step
dt=dx**2/2               # Temporal step 


# Define the simulation parameters
npasos=11000  # Number of steps
npasosrep=50  # Number of steps between representations
tpause=0.02   # Pause time between representations


# Create the mesh
mallado=np.linspace(xmin,xmax,ptos)


# Definne the potential barrier
V=np.zeros(ptos)
V[500:530]=75


# Create the propagating wave function
x0=-3.5    # Where the pulse appears
sigma=0.5  # Gaussian's wide
k0=10      # Wavenumber
fr0=np.exp(-0.5*((mallado-x0)/sigma)**2)*np.cos(k0*mallado) # Real part of the wave function
fi0=np.exp(-0.5*((mallado-x0)/sigma)**2)*np.sin(k0*mallado) # Imaginary part of the wave function


# Define a function that computes the norm
def ctenorm(fr,fi,dx):
    ctenorm=sum((fr**2+fi**2)*dx)
    return ctenorm

# Define a function that normalizes functions
def norm(fr,fi):
    frn=fr/np.sqrt(ctenorm(fr,fi,dx))
    fin=fi/np.sqrt(ctenorm(fr,fi,dx))
    return frn,fin

# Define a function with computes the second derivative
def diff2(f):
    diff2=np.zeros(ptos)                       
    diff2[0]=(f[1]+f[-1]-2*f[0])/(dx**2)               
    diff2[-1]=(f[0]+f[-2]-2*f[-1])/(dx**2)           
    diff2[1:-1]=(f[2:]+f[:-2]-2*f[1:-1])/(dx**2) 
    return diff2


# Compute the normalized functions
fr,fi=norm(fr0,fi0)

# Create the figure 
figura=plt.figure()
ejes1=figura.add_subplot(211)
funcionr,=ejes1.plot(mallado,fr,'b',label='$\psi_{Re}$(x)')
funcioni,=ejes1.plot(mallado,fi,'r',label='$\psi_{Im}$(x)')
funcionabs,=ejes1.plot(mallado,(fr**2+fi**2),'black',label='$|\psi(x)|^2$')
funcionpot,=ejes1.plot(mallado,V/max(V),'purple',ls='-')
plt.xlabel('Position',size='large')
ejes1.legend()


# Initialize the time and create the lists for the time and probabilities
t=0
T=[0]
Pa=[sum((fr**2+fi**2)[530:]*dx)*100]
Pr=[sum((fr**2+fi**2)[:500]*dx)*100]

# Create another figure to represent the probabilities
ejes2=figura.add_subplot(212)
probatravesar,=ejes2.plot(T,Pa,label='Prob of passing')
probrevotar,=ejes2.plot(T,Pr,label='Prob of bouncing')
plt.xlabel('Tiempo',size='large')
plt.ylabel('Probability (%)',size='large')
ejes2.grid()
ejes2.legend()


# Simulate it npasos times
for n in range(npasos):
    
    # Apply the wave function propagation equations 
    fi=fi+(dt/4)*diff2(fr)-(dt/2)*V*fr
    
    fr=fr-(dt/2)*diff2(fi)+dt*V*fi
    
    fi=fi+(dt/4)*diff2(fr)-(dt/2)*V*fr
    
    # Update the time
    t=t+dt
    
    
    # In this part of the code we make the position measure
    if n==10950:
        
        # Create a list to choose randomly the collapse position
        P=[]
        
        for g in range(ptos):
             
            P=P+[g]*round(((fr**2+fi**2)[g])*100)
        
        # Choose randomly one list element
        xmedida=rand.choice(P)
        
        # Reassign fi and fr values 
        fi=np.zeros(ptos)
        fr=np.zeros(ptos)
        
        fi[xmedida]=1/np.sqrt(2)
        fr[xmedida]=1/np.sqrt(2)
        
        
        
    # Plot each npasosrep iterations
    if n%npasosrep==0:
        
        # Add the time and the probabilities
        T.append(t)
        Pr.append(sum((fr**2+fi**2)[:500]*dx)*100)
        Pa.append(sum((fr**2+fi**2)[530:]*dx)*100)
        
        # Update function values
        funcionr.set_data(mallado,fr)
        funcioni.set_data(mallado,fi)
        funcionabs.set_data(mallado,np.sqrt(fr**2+fi**2))
        ejes1.set_ylim(-1.8,1.8)
        
        if n!=10950:
            
            probatravesar.set_data(T,Pa)
            probrevotar.set_data(T,Pr)
            ejes2.set_xlim(0,T[-1])
        
        plt.pause(tpause)


