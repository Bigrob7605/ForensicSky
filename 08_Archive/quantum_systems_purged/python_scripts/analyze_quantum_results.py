#!/usr/bin/env python3
"""
Analyze Quantum 50 Premium Pulsars Results
"""

import json
import numpy as np

def analyze_quantum_results():
    """Analyze the quantum tomography results"""
    
    # Load the results
    with open('quantum_50_premium_pulsars_20250905_193411.json', 'r') as f:
        data = json.load(f)
    
    print("🧠 QUANTUM 50 PREMIUM PULSARS ANALYSIS")
    print("=" * 50)
    print(f"Analysis Type: {data['analysis_type']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Number of Pulsars: {data['n_pulsars']}")
    print(f"Analysis Time: {data['analysis_time']:.2f} seconds")
    print()
    
    print("📊 QUANTUM SIGNATURE STATISTICS")
    print("-" * 40)
    signatures = data['signatures']
    print(f"High Quantum Correlations: {signatures.get('high_quantum_correlations', 0)}")
    print(f"High Entanglement Pairs: {signatures.get('high_entanglement_pairs', 0)}")
    print(f"Total Quantum Signatures: {signatures.get('total_quantum_signatures', 0)}")
    print()
    
    # Analyze the quantum kernel matrix
    kernels = np.array(data['kernels'])
    entropies = np.array(data['entropy_matrix'])
    
    print("📈 QUANTUM KERNEL MATRIX ANALYSIS")
    print("-" * 40)
    print(f"Kernel Matrix Shape: {kernels.shape}")
    print(f"Max Kernel Value: {kernels.max():.3f}")
    print(f"Min Kernel Value: {kernels.min():.3f}")
    print(f"Mean Kernel Value: {kernels.mean():.3f}")
    print(f"Std Kernel Value: {kernels.std():.3f}")
    print()
    
    print("🔬 ENTANGLEMENT ENTROPY ANALYSIS")
    print("-" * 35)
    print(f"Entropy Matrix Shape: {entropies.shape}")
    print(f"Max Entropy Value: {entropies.max():.3f}")
    print(f"Min Entropy Value: {entropies.min():.3f}")
    print(f"Mean Entropy Value: {entropies.mean():.3f}")
    print(f"Std Entropy Value: {entropies.std():.3f}")
    print()
    
    # Find the most interesting pairs
    print("🌟 MOST INTERESTING QUANTUM PAIRS")
    print("-" * 40)
    
    # Find the most interesting pairs by looking at the kernel and entropy matrices
    n_pulsars = len(data['pulsar_ids'])
    
    # High kernel, low entropy (classical correlation)
    high_kernel_mask = kernels > 0.5
    low_entropy_mask = entropies < 0.1
    classical_pairs = np.where(high_kernel_mask & low_entropy_mask)
    
    print("Classical Correlation Pairs (High Kernel, Low Entropy):")
    count = 0
    for i, (row, col) in enumerate(zip(classical_pairs[0], classical_pairs[1])):
        if row < col and count < 5:  # Avoid duplicates and limit to 5
            pulsar1 = data['pulsar_ids'][row]
            pulsar2 = data['pulsar_ids'][col]
            print(f"  {pulsar1} ↔ {pulsar2}: K={kernels[row,col]:.3f}, S={entropies[row,col]:.3f}")
            count += 1
    
    print()
    
    # High entropy, low kernel (quantum entanglement)
    high_entropy_mask = entropies > 0.5
    low_kernel_mask = kernels < 0.3
    quantum_pairs = np.where(high_entropy_mask & low_kernel_mask)
    
    print("Quantum Entanglement Pairs (High Entropy, Low Kernel):")
    count = 0
    for i, (row, col) in enumerate(zip(quantum_pairs[0], quantum_pairs[1])):
        if row < col and count < 5:  # Avoid duplicates and limit to 5
            pulsar1 = data['pulsar_ids'][row]
            pulsar2 = data['pulsar_ids'][col]
            print(f"  {pulsar1} ↔ {pulsar2}: K={kernels[row,col]:.3f}, S={entropies[row,col]:.3f}")
            count += 1
    
    print()
    print("🎯 ANALYSIS COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    analyze_quantum_results()
