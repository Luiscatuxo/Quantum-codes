# This code simulates a simple, three-mode Boson Sampling quantum optical circuit.
# By injecting two single photons on mode 0 and 1 in the circuit, it simulates the
# experiment and computes the probability of each state.

# Import all the necessary libraries and elements
import perceval as pcvl
from perceval import PS, BS, PERM
from perceval.algorithm import Sampler
import numpy as np
import matplotlib.pyplot as plt

# Create a circuit with 3 modes
circuit = pcvl.Circuit(3)
circuit.add(0, PERM([1, 2, 0]))
circuit.add(1, BS.H())
circuit.add(0, PS(np.pi/2))
circuit.add(0, PERM([1, 0]))
circuit.add(1, BS.H())

pcvl.pdisplay(circuit)                                 # Show the circuit

input_state = pcvl.BasicState([1,1,0])                 # Initial state: one photon on first modes = |1,1,0> (in Fock basis)

# Preparate the simulation
processor = pcvl.Processor('SLOS', circuit)
processor.with_input(input_state)

# Make the simulation
sampler = Sampler(processor)
samples = sampler.sample_count(10_000)['results']      # Ask to generate 10k samples, and get back only the raw results

# Compute the experimental probabilities
total = sum(samples.values())                          # Sum all the counts = 10_000
probsexp = {state: count/total for state, count in samples.items()}

# Extract both the final states and their numerical experimental probabilities
states = [str(s) for s in probsexp.keys()]
values = list(probsexp.values())

# Make the plot
plt.figure()
plt.bar(range(len(values)), values)
plt.xticks(range(len(values)), states)
plt.ylabel('Probability')
plt.title('Boson Sampling')
plt.show()