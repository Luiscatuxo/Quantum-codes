# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:55:55 2025

@author: Iker León
"""
# This Python code implements QAOA (Quantum Approximate Optimization Algorithm)
# to solve the Longest Cycle Problem on 20 different graphs. It first defines the graphs
# and builds a corresponding cost Hamiltonian using Pauli operators, including penalties to enforce valid cycles.
# It then creates a QAOA ansatz circuit, transpiles it for an IBM Quantum backend, and optimizes the parameters
# using classical optimization (COBYLA). After running the optimized circuit on a quantum simulator or hardware,
# it samples the resulting bitstrings, identifies the best solution, and analyzes it: counts active edges, vertex degrees, cycle length, and connectivity.
# Finally, the code visualizes the graphs and cycles, saves images, and stores all results in an Excel file.

# QAOA para el Ciclo Más Largo - Versión con 20 Grafos y Conexión Cuántica
import pandas as pd
import rustworkx as rx
import numpy as np
import itertools
import os
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Sampler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

########## CONFIGURACIÓN INICIAL ##########

save_path = 'C:\\Users\\DELL LATITUDE 7480\\Desktop\\IBM course\\Runeada'

# Crear el directorio si no existe
os.makedirs(save_path, exist_ok=True)

# Configura tu cuenta de IBM Quantum
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(min_num_qubits=127, operational=True)

########## DEFINICIÓN DE GRAFOS ##########

# Definimos 20 grafos para probar el algoritmo
test_graphs = [
    # 1. Triángulo (ciclo de 3 nodos)
    ([0, 1, 2], [1, 2, 0]),
    
    # 2. Cuadrado (ciclo de 4 nodos)
    ([0, 1, 2, 3], [1, 2, 3, 0]),
    
    # 3. Cuadrado con diagonal
    ([0, 1, 2, 3, 0], [1, 2, 3, 0, 2]),
    
    # 4. Pentágono (ciclo de 5 nodos)
    ([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]),
    
    # 5. Hexágono (ciclo de 6 nodos)
    ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]),
    
    # 6. Hexágono con 3 diagonales
    ([0, 1, 2, 3, 4, 5, 0, 2, 4], [1, 2, 3, 4, 5, 0, 2, 4, 0]),
    
    # 7. Dos triángulos conectados
    ([0, 1, 2, 2, 3, 4, 5], [1, 2, 0, 3, 4, 5, 3]),
    
    # 8. Grafo completo K4
    ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]),
    
    # 9. Estrella de 4 puntas
    ([0, 0, 0, 0], [1, 2, 3, 4]),
    
    # 10. Grafo lineal de 5 nodos
    ([0, 1, 2, 3], [1, 2, 3, 4]),
    
    # 11. Cubo
    ([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]),
    
    # 12. Ciclo de 7 nodos
    ([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]),
    
    # 13. Ciclo de 8 nodos con diagonales
    ([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6], [1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 0]),
    
    # 14. Grafo bipartito completo K3,3
    ([0, 0, 0, 1, 1, 1, 2, 2, 2], [3, 4, 5, 3, 4, 5, 3, 4, 5]),
    
    # 15. Grafo rueda de 5 nodos
    ([0, 0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 2, 3, 4]),
    
    # 16. Grafo de Petersen
    ([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9], [1, 4, 5, 2, 6, 3, 7, 4, 8, 0, 9, 7, 8, 9, 5, 6]),
    
    # 17. Grafo con dos componentes conexos
    ([0, 1, 2, 4, 5], [1, 2, 0, 5, 4]),
    
    # 18. Grafo con nodo aislado
    ([0, 1, 2, 3], [1, 2, 3, 4]),  # Nodo 4 está aislado
    
    # 19. Grafo con puente
    ([0, 1, 2, 3, 4, 2], [1, 2, 3, 4, 5, 5]),
    
    # 20. Grafo completo K5
    ([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
]

########## FUNCIONES AUXILIARES ##########

def build_longest_cycle_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Hamiltoniano para el problema del ciclo más largo."""
    pauli_list = []
    edge_count = len(graph.edge_list())
    node_to_edges = {node: [] for node in graph.nodes()}

    for i, (u, v) in enumerate(graph.edge_list()):
        node_to_edges[u].append(i)
        node_to_edges[v].append(i)

    # Término de costo: maximizar aristas activas
    for i in range(edge_count):
        paulis = ["I"] * edge_count
        paulis[i] = "Z"
        pauli_list.append(("".join(paulis)[::-1], -1.0))

    # Restricción: solo grados 0 o 2
    penalty_strength = 100.0
    for node, edges in node_to_edges.items():
        for k in range(1, len(edges) + 1):
            if k == 2:
                continue
            for combo in itertools.combinations(edges, k):
                paulis = ["I"] * edge_count
                for edge_idx in combo:
                    paulis[edge_idx] = "Z"
                pauli_list.append(("".join(paulis)[::-1], penalty_strength))

    # Prevención de subciclos
    for i, edge1 in enumerate(graph.edge_list()):
        u1, v1 = edge1
        for j, edge2 in enumerate(graph.edge_list()):
            if i < j:
                u2, v2 = edge2
                if {u1, v1}.intersection({u2, v2}):
                    continue
                if not (rx.has_path(graph, u1, u2) or rx.has_path(graph, u1, v2) or
                       rx.has_path(graph, v1, u2) or rx.has_path(graph, v1, v2)):
                    paulis = ["I"] * edge_count
                    paulis[i] = "Z"
                    paulis[j] = "Z"
                    pauli_list.append(("".join(paulis)[::-1], penalty_strength))

    return pauli_list

def create_graph(source_list, target_list):
    """Crea un grafo a partir de listas de nodos fuente y destino."""
    prepared_graph = rx.PyGraph()
    n = max(max(source_list), max(target_list)) + 1
    prepared_graph.add_nodes_from(range(n))
    elist = [(s, t, 1.0) for s, t in zip(source_list, target_list)]
    prepared_graph.add_edges_from(elist)
    return prepared_graph

def to_bitstring(number: int, length: int) -> list:
    """Convierte un número a un bitstring con la longitud dada"""
    return [int(bit) for bit in format(number, f"0{length}b")]

def analyze_solution(graph, bitstring, graph_id):  
    """Analiza la solución del ciclo hamiltoniano."""
    edge_list = list(graph.edge_list())
    total_edges = len(edge_list)
    
    active_edges = [(u, v, 1.0) for (u, v), bit in zip(edge_list, bitstring) if bit == 1]
    inactive_edges = [(u, v, 1.0) for (u, v), bit in zip(edge_list, bitstring) if bit == 0]
    active_edge_count = len(active_edges)
    
    subG = rx.PyGraph()
    subG.add_nodes_from(graph.nodes())
    subG.add_edges_from(active_edges)
    
    degrees = {node: subG.degree(node) for node in subG.nodes()}
    
    degree_counts = {0: 0, 2: 0, 'other': 0}
    for degree in degrees.values():
        if degree == 0:
            degree_counts[0] += 1
        elif degree == 2:
            degree_counts[2] += 1
        else:
            degree_counts['other'] += 1
    
    cycle_length = active_edge_count
    components = list(rx.connected_components(subG))
    is_single_cycle = (len(components) == 1 and degree_counts['other'] == 0)
    
    print("\n" + "="*50)
    print(f"ANÁLISIS DE LA SOLUCIÓN PARA GRAFO {graph_id}")
    print("="*50)
    print(f"Total de vértices: {len(graph.nodes())}")
    print(f"Total de aristas en el grafo: {total_edges}")
    print(f"Aristas activas en la solución: {active_edge_count}")
    print(f"Aristas inactivas en la solución: {total_edges - active_edge_count}")
    print("\nDistribución de grados de vértices:")
    print(f"  Vértices con grado 0: {degree_counts[0]}")
    print(f"  Vértices con grado 2: {degree_counts[2]}")
    print(f"  Vértices con otros grados: {degree_counts['other']}")
    print(f"\nLongitud del ciclo (aristas activas): {cycle_length}")
    print(f"¿Es un ciclo válido?: {'Sí' if is_single_cycle else 'No'}")
    
    if degree_counts['other'] > 0:
        print("\nVértices con problemas (grado ≠ 0 o 2):")
        for node, degree in degrees.items():
            if degree != 0 and degree != 2:
                print(f"  Vértice {node}: grado {degree}")
    
    if len(components) > 1:
        print(f"\n¡Advertencia! Hay {len(components)} componentes conexos:")
        for i, comp in enumerate(components, 1):
            print(f"  Componente {i}: {len(comp)} vértices")
    
    print("="*50 + "\n")
    
    return {
        'graph_id': graph_id,
        'total_vertices': len(graph.nodes()),
        'total_edges': total_edges,
        'active_edges': active_edge_count,
        'inactive_edges': total_edges - active_edge_count,
        'degree_0': degree_counts[0],
        'degree_2': degree_counts[2],
        'degree_other': degree_counts['other'],
        'cycle_length': cycle_length,
        'is_valid_cycle': is_single_cycle,
        'components': len(components)
    }

def plot_cycle_on_graph(G, bitstring, title, filename):
    """Visualiza el ciclo encontrado en el grafo y lo guarda como imagen."""
    edge_list = list(G.edge_list())
    active_edges = [(u, v, 1.0) for (u, v), bit in zip(edge_list, bitstring) if bit == 1]
    
    pos = rx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    rx.visualization.mpl_draw(G, pos=pos, ax=ax, node_size=300, 
                             node_color="lightgrey", with_labels=True, 
                             edge_color="lightgrey", width=2)
    
    if active_edges:
        subG = rx.PyGraph()
        subG.add_nodes_from(G.nodes())
        subG.add_edges_from(active_edges)
        rx.visualization.mpl_draw(subG, pos=pos, ax=ax, node_size=300,
                                 node_color="orange", edge_color="red", width=4)
    
    plt.title(title)
    # Guardar la figura
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
    plt.close()  # Cerrar la figura para liberar memoria

########## EJECUCIÓN PRINCIPAL ##########

# DataFrame para resultados
results_df = pd.DataFrame(columns=[
    'graph_id', 'total_vertices', 'total_edges', 'active_edges', 
    'inactive_edges', 'degree_0', 'degree_2', 'degree_other',
    'cycle_length', 'is_valid_cycle', 'components'
])

# Función de costo para QAOA
def cost_function(params, ansatz, hamiltonian, estimator):
    """Función de costo para la optimización de QAOA"""
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)       
    job = estimator.run([pub])
    result = job.result()[0]
    cost = result.data.evs
    return cost

# Función para encontrar la mejor solución
def best_solution(samples, hamiltonian):
    """Encuentra la solución con menor costo"""
    min_sol = None
    min_cost = np.inf
    
    for sample in samples.keys():
        cost = calculate_cost(sample, hamiltonian)
        if cost < min_cost:
            min_cost = cost
            min_sol = sample
    
    return min_sol

# Calcular costos
_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)

def calculate_cost(state: int, observable: SparsePauliOp) -> complex:
    """Calcula el costo de un estado medido"""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced]).real

########## PROCESAMIENTO DE GRAFOS ##########

for i, (sources, targets) in enumerate(test_graphs):
    print(f"\n{'='*30} Procesando Grafo {i+1} {'='*30}")
    
    # Crear el grafo
    prepared_graph = create_graph(sources, targets)
    edge_count = len(prepared_graph.edge_list())
    print(f"Grafo {i+1}: {len(prepared_graph.nodes())} nodos, {edge_count} aristas")
    
    # Visualizar grafo original y guardarlo
    plt.figure(figsize=(6, 6))
    rx.visualization.mpl_draw(prepared_graph, node_size=300, with_labels=True, 
                            node_color='skyblue', edge_color='gray')
    plt.title(f"Grafo {i+1} Original")
    original_graph_filename = f"Grafo_{i+1}_Original.png"
    plt.savefig(os.path.join(save_path, original_graph_filename), bbox_inches='tight')
    plt.close()
    
    # Construir Hamiltoniano
    cycle_paulis = build_longest_cycle_paulis(prepared_graph)
    cost_hamiltonian = SparsePauliOp.from_list(cycle_paulis)
    
    # Crear circuito QAOA
    qaoa_circuit = QAOAAnsatz(cost_hamiltonian, reps=2)
    qaoa_circuit.measure_all()
    
    # Transpilar el circuito
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit_max_cycle = pm.run(qaoa_circuit)
    
    # Ejecutar optimización
    print(f"Ejecutando QAOA para Grafo {i+1}...")
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.resilience_level = 1
        
        initial_params = np.random.rand(4)  # 2*reps parámetros
        
        result = minimize(
            cost_function,
            initial_params,
            args=(candidate_circuit_max_cycle, cost_hamiltonian, estimator),
            method="COBYLA",
            options={"maxiter": 1},
            #tol=1e-3
        )
    
    # Obtener la mejor solución
    optimized_circuit_max_cycle = candidate_circuit_max_cycle.assign_parameters(result.x)
    
    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        sampler.options.default_shots = 10000
        
        # Ejecutar circuito optimizado
        pub = (optimized_circuit_max_cycle, )
        job = sampler.run([pub], shots=int(1e4))
        counts_int = job.result()[0].data.meas.get_int_counts()
        shots = sum(counts_int.values())
        final_distribution = {key: val/shots for key, val in counts_int.items()}
    
    # Encontrar y analizar la mejor solución
    best_sol = best_solution(final_distribution, cost_hamiltonian)
    best_solution_string = to_bitstring(int(best_sol), edge_count)
    best_solution_string.reverse()
    
    # Analizar y guardar resultados
    analysis_results = analyze_solution(prepared_graph, best_solution_string, i+1)
    results_df = pd.concat([results_df, pd.DataFrame([analysis_results])], ignore_index=True)
    
    # Visualizar solución y guardarla
    solution_filename = f"Grafo_{i+1}_Solucion_QAOA.png"
    plot_cycle_on_graph(prepared_graph, best_solution_string, 
                       f"Solución QAOA para Grafo {i+1}", solution_filename)

# Guardar resultados en Excel
results_excel_path = os.path.join(save_path, "resultados_qaoa_20grafos.xlsx")
results_df.to_excel(results_excel_path, index=False)
print(f"\nResultados guardados en '{results_excel_path}'")

# Mostrar resumen
print("\nRESUMEN DE RESULTADOS:")
print(results_df[['graph_id', 'cycle_length', 'is_valid_cycle', 'components']])