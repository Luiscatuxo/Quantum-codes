# This code simulates a Ralph's et al. CNOT gate between two qubits using dual-rail encoding.
# By selecting the initial logic state |1,0>, it simulates the implementation of the gate, 
# measures the success probability and compares it to the theoretical value for different number of shots.

# Import all the necessary libraries and elements
import perceval as pcvl
from perceval import PS, BS, PERM
from perceval.algorithm import Sampler
import numpy as np
import matplotlib.pyplot as plt
print(pcvl.__version__)
# Create a list with all number of shots the experiment is repeated 
shots = np.linspace(5000, 10000000, 50)

# Create the circuit with 2 modes for the ancilla photons (modes 1 and 5) qubit 1: modes 0 and 2, qubit 2: modes 3 and 4
Ralph = pcvl.Circuit(6)
Ralph.add(0, BS.H(theta=2*np.arccos(np.sqrt(1/3))))                # Center top beam splitter
Ralph.add(3, BS.H())                                               # Central left beam splitter
Ralph.add(4, BS.H(theta=2*np.arccos(np.sqrt(1/3))))                # Bottom beam splitter
Ralph.add(2, PERM([1, 0]))                                         
Ralph.add(2, BS.H(theta=2*np.arccos(np.sqrt(1/3))))                # Central beam splitter
Ralph.add(2, PERM([1, 0]))
Ralph.add(3, BS.H())                                               # Central right beam splitter

pcvl.pdisplay(Ralph)                                               # Show the circuit

# Define the intial state with 2 photons (qubit 1 in |1> and qubit 2 in |0> = |0, 0, 1, 1, 0, 0> in Fock basis )
input_state = pcvl.BasicState([0, 0, 1, 1, 0, 0])

# Preparate the simulation
processor = pcvl.Processor('SLOS', Ralph)
processor.with_input(input_state)

# Make the simulation
sampler = Sampler(processor)

# Create a list to store the success probability
P = []
for counts in shots:
    
    samples = sampler.sample_count(counts)['results']       # Ask to generate 'counts' samples, and get back only the raw results

    total = sum(samples.values())                           # Sum all counts = 10_000

    # Extract the probability numbers
    probsexp = {state: count/total for state, count in samples.items()}
    valores = list(probsexp.values())

    # Extract the probability of the desired final state |0, 0, 1, 0, 1, 0> = |1,1> in logical basis
    psuccess = valores[2]
    P.append(psuccess)

# Make the plot to compare
plt.figure()
plt.plot(shots, P)
plt.xlabel('Number of shots')
plt.xscale('log')
plt.ylabel('Success probability')
plt.axhline(y=1/9, label='Theoretical probability = 1/9', color='orange')
plt.legend()
plt.show()
