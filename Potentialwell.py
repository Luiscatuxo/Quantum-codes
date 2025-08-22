# -*- coding: utf-8 -*-
"""
Electron in a potential well
Luis Fernández-Catuxo Ortiz
uo283944
"""

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the system's data
a=1e-10                  # Well's width(m)
V0=244                   # Potential out of the well (eV)
c=3e8                    # Light velocity (m/s^2)
m=0.511e6/c**2           # Electron mass (eV/c^2)
h=6.582e-16              # Planck's cte (eV·s)
k=(2*m*a**2*V0)/h**2     # Cte
alfa0=np.pi**2/(2*k)     # Intial value for alpha

# Define the simulation data
ptos=500                       # Half the number of the mesh points 
mallado=np.linspace(0,4,ptos)  # Mesh of half well
niter=50                       # Number of iterations to compute the energie

# Define C function
def C(mallado,alfa):
    C=[]
    for i in mallado:
        if i<0.5:
            c=-k*alfa
            C.append(c)
        else:
            c=k*(1-alfa)
            C.append(c)
    return C

# Define a function to compute the wafe functions 
def psi(alfa,par):
    
    # Create the wave functions
    psi=np.zeros(ptos)
    diffpsi=np.zeros(ptos)
    
    # Establish the initial conditions depending if it is odd or even
    if par==True:
        psi[0]=1
        diffpsi[0]=0
        
    if par==False:
        psi[0]=0
        diffpsi[0]=1
    
    # Establish the equations to compute psi on every mesh point
    for j in range(ptos-1):
        
        # Compute the wave function
        psi[j+1]=psi[j]+diffpsi[j]*(mallado[j+1]-mallado[j])
        
        # Compute the derivative of the wave function
        diffpsi[j+1]=diffpsi[j]+C(mallado,alfa)[j]*psi[j]*(mallado[j+1]-mallado[j])
    return psi
   

# Divide the interval between alpha0 and 1 in subintervals 
ninterval=3
alfas=np.linspace(alfa0,1,ninterval)

# Define a function which computes if there is a sign change in a subinterval of alphas
def signo(psi,alfa1,alfa2,par):
    return psi(alfa1,par)[-1]*psi(alfa2,par)[-1]


# Define a function which computes the values of alpha for odd and evens
def Energia(alfas,par):
    
    # Create a list for adding the final alpha values
    Alfa=[]
    
    # Iterate for each interval
    for p in range(ninterval-1):
        
        # Assign the edges of the interval to two variables
        alfa1=alfas[p]
        alfa2=alfas[p+1]
        
        # Check if there is a sign change inside the interval
        if signo(psi,alfa1,alfa2,par)<0:
            
            # Repeat the process niter times
            for y in range(niter):
                
                # Compute the mean value of the interval
                alfamedio=(alfa1+alfa2)/2
                
                # If there is a change o sign in the first half we update
                if signo(psi,alfa1,alfamedio,par)<0:
                    alfa2=alfamedio
                
                # If there is a change o sign in the second half we update
                if signo(psi,alfamedio,alfa2,par)<0:
                    alfa1=alfamedio
                    
            Alfa.append(alfamedio)
    return Alfa

# Create the wave functions and the mesh for all the well
malladototal=np.zeros(2*ptos)
psi1=np.zeros(2*ptos)
psi2=np.zeros(2*ptos)
psi3=np.zeros(2*ptos)

# Define the mesh for all the well
malladototal[0:ptos]=-np.flip(mallado)
malladototal[ptos:]=mallado

# Define two useful variables
g=Energia(alfas,True)[0]
h=Energia(alfas,True)[1]

# Compute the wave functions in the other half 
psi1[0:ptos]=np.flip(psi(g,True))
psi1[ptos:]=psi(g,True)
    
psi2[0:ptos]=np.flip(psi(h,True))
psi2[ptos:]=psi(h,True)

psi3[0:ptos]=-np.flip(psi(Energia(alfas,False)[0],False))
psi3[ptos:]=psi(Energia(alfas,False)[0],False)


# Create the figure and represent the wave functions
figura=plt.figure()
ejes=figura.add_subplot(111)
funciononda1,=ejes.plot(malladototal,psi1,'b',label='$\psi_1(x)$')
funciononda2,=ejes.plot(malladototal,psi2,'r',label='$\psi_2(x)$')
funciononda3,=ejes.plot(malladototal,psi3,'black',label='$\psi_3(x)$')
plt.xlabel('Position')
ejes.legend()
ejes.grid()
