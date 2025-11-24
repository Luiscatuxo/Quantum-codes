# This code simulates a two-mode quantum optical circuit to illustrate the Hong–Ou–Mandel (HOM) effect.
# By varying a phase, it computes the probabilities of different output states and plots them, showing 
# how photon bunching depends on the phase shift.

# Import all the necessary libraries and elements
import perceval as pcvl
from perceval import PS, BS
from perceval.algorithm import Sampler
import numpy as np
import matplotlib.pyplot as plt

# Create a list with different angles of the phase shifter
angles = np.linspace(0.1, 3.1, 100)

# Initial state: one photon on each mode = |1,1> (in Fock basis)
input_state = pcvl.BasicState("|1,1>")

# Create lists to store the probabilities of each outcome state
P_20 = []
P_11 = []
P_02 = []

# Iterate over the angles
for phi in angles:
    circuit = pcvl.Circuit(2)
    circuit.add(0, BS.H())        # Beam splitter
    circuit.add(1, PS(phi))       # Phase in mode 1
    circuit.add(0, BS.H())        # Beam splitter

    if phi == 0.1:
        pcvl.pdisplay(circuit)   # Show the circuit

    # Preparate the simulation
    processor = pcvl.Processor('SLOS', circuit)
    processor.with_input(input_state)

    # Make the simulation
    sampler = Sampler(processor)
    samples = sampler.sample_count(10_000)['results']         # Ask to generate 10k samples, and get back only the raw results
    probstheo = sampler.probs()['results']                    # Ask for the exact probabilities
    total = sum(samples.values())                             # Sum all the counts = 10_000
    print(probstheo)

    # Extract the probability numbers so we can plot them
    probsexp = {state: count/total for state, count in samples.items()}
    valores = list(probsexp.values())
    
    # Append the probabilities to the lists
    P_20.append(valores[0])
    P_11.append(valores[1])
    P_02.append(valores[2])

# Make the plot
plt.plot(angles/np.pi, P_20, label=r'$P_{|2,0\rangle}$')
plt.plot(angles/np.pi, P_11, label=r'$P_{|1,1\rangle}$')
plt.plot(angles/np.pi, P_02, label=r'$P_{|0,2\rangle}$')
plt.plot(angles/np.pi, [x + y + z for x,y,z in zip(P_20, P_11, P_02)],
         label='Prob sum', color='black', linestyle='--')

plt.xlabel(r'$\phi/\pi$')
plt.ylabel('Probability')
plt.legend()
plt.show()