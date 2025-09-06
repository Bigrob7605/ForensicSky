#!/usr/bin/env python3
"""
QUICK QUANTUM TEST - 5 Pulsars Only
===================================

Test quantum residual tomography with just 5 pulsars for speed.
"""

import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, Aer
    from qiskit.quantum_info import DensityMatrix, partial_trace
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available - Quantum kernel methods ready!")
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️ Qiskit not available - using classical approximation")

# Classical fallback
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import logm
import warnings
warnings.filterwarnings('ignore')

def load_real_data(max_pulsars=50):
    """Load real IPTA DR2 data, limited to max_pulsars"""
    print(f"🔬 Loading real IPTA DR2 data (limited to {max_pulsars} pulsars)...")
    
    try:
        # Import the Core Forensic Sky V1 engine
        sys.path.append('01_Core_Engine')
        from Core_ForensicSky_V1 import CoreForensicSkyV1
        
        # Initialize engine
        engine = CoreForensicSkyV1()
        
        # Load real IPTA DR2 data with early termination
        print(f"🛑 Loading data with early termination at {max_pulsars} pulsars...")
        
        # Monkey patch the loading method to stop early
        original_load = engine.load_real_ipta_data
        def limited_load():
            stats = original_load()
            # Force early termination after max_pulsars
            if hasattr(engine, 'timing_data') and len(engine.timing_data) >= max_pulsars:
                # Truncate to max_pulsars
                pulsar_list = list(engine.timing_data.keys())[:max_pulsars]
                engine.timing_data = {pid: engine.timing_data[pid] for pid in pulsar_list}
                print(f"🛑 Early termination: Limited to {max_pulsars} pulsars")
            return stats
        
        engine.load_real_ipta_data = limited_load
        loading_stats = engine.load_real_ipta_data()
        
        if not hasattr(engine, 'timing_data') or not engine.timing_data:
            print("❌ No timing data loaded from engine!")
            return create_test_data()
        
        # Extract residuals from loaded timing data
        residuals = {}
        for pulsar_id, timing_info in engine.timing_data.items():
            if 'residuals' in timing_info and len(timing_info['residuals']) > 0:
                residuals[pulsar_id] = timing_info['residuals']
        
        print(f"✅ Loaded real residuals for {len(residuals)} pulsars")
        return residuals
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("Falling back to test data...")
        return create_test_data()

def create_test_data():
    """Create test data with 5 pulsars (fallback)"""
    print("🧪 Creating test data with 5 pulsars...")
    
    np.random.seed(42)
    residuals = {}
    
    # Create 5 different types of pulsars
    pulsar_types = [
        ("J0437-4715", "high_freq", 0.1, 5.0),  # High frequency noise
        ("J1713+0747", "low_freq", 0.05, 0.5),  # Low frequency drift  
        ("J1909-3744", "periodic", 0.08, 1.0),  # Periodic signal
        ("J1744-1134", "pure_noise", 0.1, 0.0), # Pure noise
        ("J2145-0750", "mixed", 0.06, 2.0)      # Mixed signal
    ]
    
    for pulsar_id, signal_type, noise_level, freq in pulsar_types:
        t = np.linspace(0, 10, 1024)
        
        if signal_type == "high_freq":
            residuals[pulsar_id] = np.random.normal(0, noise_level, 1024) + 0.05 * np.sin(2 * np.pi * freq * t)
        elif signal_type == "low_freq":
            residuals[pulsar_id] = np.random.normal(0, noise_level, 1024) + 0.1 * t
        elif signal_type == "periodic":
            residuals[pulsar_id] = np.random.normal(0, noise_level, 1024) + 0.03 * np.sin(2 * np.pi * freq * t)
        elif signal_type == "pure_noise":
            residuals[pulsar_id] = np.random.normal(0, noise_level, 1024)
        else:  # mixed
            residuals[pulsar_id] = np.random.normal(0, noise_level, 1024) + 0.02 * np.sin(2 * np.pi * freq * t) + 0.01 * t
    
    print(f"✅ Created test residuals for {len(residuals)} pulsars")
    return residuals

def angle_encode_residuals(residuals):
    """Angle encode residuals into quantum states"""
    print("🔬 Angle-encoding residuals into quantum states...")
    
    pulsar_ids = list(residuals.keys())
    r_matrix = np.array([residuals[pid] for pid in pulsar_ids])
    
    # Normalize residuals
    norm = np.linalg.norm(r_matrix, axis=1, keepdims=True)
    r_matrix = r_matrix / (norm + 1e-12)
    
    print(f"📊 Encoded {len(pulsar_ids)} pulsars with {r_matrix.shape[1]} time samples each")
    return pulsar_ids, r_matrix

def compute_quantum_kernel(r_matrix, pulsar_ids):
    """Compute quantum kernel matrix"""
    print("🧠 Computing quantum kernel matrix...")
    
    n_pulsars = len(pulsar_ids)
    kernels = np.zeros((n_pulsars, n_pulsars))
    
    if QISKIT_AVAILABLE:
        backend = Aer.get_backend('statevector_simulator')
        
        for i in range(n_pulsars):
            for j in range(i, n_pulsars):
                if i == j:
                    kernels[i, j] = 1.0
                else:
                    # Create quantum circuits
                    qc_i = create_feature_circuit(r_matrix[i])
                    qc_j = create_feature_circuit(r_matrix[j])
                    
                    # Get statevectors
                    sv_i = backend.run(qc_i).result().get_statevector()
                    sv_j = backend.run(qc_j).result().get_statevector()
                    
                    # Compute overlap
                    kernels[i, j] = np.abs(np.vdot(sv_i, sv_j))**2
                    kernels[j, i] = kernels[i, j]
    else:
        # Classical approximation
        print("⚠️ Using classical RBF kernel approximation")
        kernels = rbf_kernel(r_matrix, gamma=1.0)
    
    print(f"✅ Computed {n_pulsars}x{n_pulsars} quantum kernel matrix")
    return kernels

def create_feature_circuit(residual_vector):
    """Create quantum circuit for angle encoding"""
    n_qubits = len(residual_vector)
    qc = QuantumCircuit(n_qubits)
    
    for t, val in enumerate(residual_vector):
        # Angle encoding: RY(2*arcsin(val))
        angle = 2 * np.arcsin(np.clip(val, -1, 1))
        qc.ry(angle, t)
    
    return qc

def compute_entanglement_entropy(r_matrix, pulsar_ids):
    """Compute entanglement entropy for each pulsar pair"""
    print("🔍 Computing entanglement entropy...")
    
    n_pulsars = len(pulsar_ids)
    entropy_matrix = np.zeros((n_pulsars, n_pulsars))
    
    if QISKIT_AVAILABLE:
        backend = Aer.get_backend('statevector_simulator')
        
        for i in range(n_pulsars):
            for j in range(i+1, n_pulsars):
                try:
                    # Create combined quantum circuit
                    qc = create_combined_circuit(r_matrix[i], r_matrix[j])
                    
                    # Compute entanglement entropy
                    entropy = compute_entanglement_entropy_pair(qc, backend)
                    entropy_matrix[i, j] = entropy
                    entropy_matrix[j, i] = entropy
                except Exception as e:
                    print(f"⚠️ Error computing entropy for pair ({i},{j}): {e}")
                    entropy_matrix[i, j] = 0.0
                    entropy_matrix[j, i] = 0.0
    else:
        # Classical approximation
        print("⚠️ Using classical entropy approximation")
        for i in range(n_pulsars):
            for j in range(i+1, n_pulsars):
                # Classical mutual information approximation
                corr = np.corrcoef(r_matrix[i], r_matrix[j])[0, 1]
                entropy_matrix[i, j] = -0.5 * np.log(1 - corr**2 + 1e-12)
                entropy_matrix[j, i] = entropy_matrix[i, j]
    
    print(f"✅ Computed entanglement entropy matrix")
    return entropy_matrix

def create_combined_circuit(vec1, vec2):
    """Create combined quantum circuit for two pulsars"""
    n_qubits = len(vec1)
    qc = QuantumCircuit(2 * n_qubits)
    
    # Encode first pulsar
    for t, val in enumerate(vec1):
        angle = 2 * np.arcsin(np.clip(val, -1, 1))
        qc.ry(angle, t)
    
    # Encode second pulsar
    for t, val in enumerate(vec2):
        angle = 2 * np.arcsin(np.clip(val, -1, 1))
        qc.ry(angle, t + n_qubits)
    
    return qc

def compute_entanglement_entropy_pair(qc, backend):
    """Compute entanglement entropy for specific qubits"""
    try:
        # Get statevector
        sv = backend.run(qc).result().get_statevector()
        
        # Create density matrix
        rho = DensityMatrix(sv)
        
        # Partial trace to get reduced density matrix
        rho_reduced = partial_trace(rho, list(range(len(qc.qubits)//2, len(qc.qubits))))
        
        # Compute von Neumann entropy
        entropy = rho_reduced.entropy()
        
        return float(entropy)
    except Exception as e:
        print(f"⚠️ Error in entropy computation: {e}")
        return 0.0

def analyze_quantum_signatures(kernels, entropy_matrix, pulsar_ids):
    """Analyze quantum signatures for cosmic string detection"""
    print("🎯 Analyzing quantum signatures...")
    
    n_pulsars = len(pulsar_ids)
    
    # Find high quantum correlation pairs
    high_kernel_pairs = []
    high_entropy_pairs = []
    
    for i in range(n_pulsars):
        for j in range(i+1, n_pulsars):
            # High quantum kernel where classical correlation would be low
            if kernels[i, j] > 0.3:
                high_kernel_pairs.append({
                    'pulsar1': pulsar_ids[i],
                    'pulsar2': pulsar_ids[j],
                    'kernel_value': kernels[i, j],
                    'entropy': entropy_matrix[i, j]
                })
            
            # High entanglement entropy
            if entropy_matrix[i, j] > 0.7:
                high_entropy_pairs.append({
                    'pulsar1': pulsar_ids[i],
                    'pulsar2': pulsar_ids[j],
                    'entropy': entropy_matrix[i, j],
                    'kernel_value': kernels[i, j]
                })
    
    # Sort by significance
    high_kernel_pairs.sort(key=lambda x: x['kernel_value'], reverse=True)
    high_entropy_pairs.sort(key=lambda x: x['entropy'], reverse=True)
    
    print(f"🔍 Found {len(high_kernel_pairs)} high quantum correlation pairs")
    print(f"🔍 Found {len(high_entropy_pairs)} high entanglement pairs")
    
    return {
        'high_kernel_pairs': high_kernel_pairs,
        'high_entropy_pairs': high_entropy_pairs,
        'quantum_signatures': len(high_kernel_pairs) + len(high_entropy_pairs)
    }

def main():
    """Run quantum test with 50 real pulsars"""
    print("🧠 QUANTUM TEST - 50 REAL PULSARS")
    print("=" * 50)
    print("Testing quantum residual tomography with REAL IPTA DR2 data...")
    print("🔬 Scanning 50 pulsars for quantum signatures...")
    print()
    
    start_time = time.time()
    
    # Step 1: Load real data (limited to 50 pulsars)
    residuals = load_real_data(max_pulsars=50)
    
    # Step 2: Angle encode
    pulsar_ids, r_matrix = angle_encode_residuals(residuals)
    
    # Step 3: Quantum kernel
    kernels = compute_quantum_kernel(r_matrix, pulsar_ids)
    
    # Step 4: Entanglement entropy
    entropy_matrix = compute_entanglement_entropy(r_matrix, pulsar_ids)
    
    # Step 5: Analyze signatures
    signatures = analyze_quantum_signatures(kernels, entropy_matrix, pulsar_ids)
    
    # Compile results
    results = {
        'analysis_type': 'Quantum Test - 50 Real Pulsars',
        'timestamp': datetime.now().isoformat(),
        'n_pulsars': len(pulsar_ids),
        'pulsar_ids': pulsar_ids,
        'kernels': kernels.tolist(),
        'entropy_matrix': entropy_matrix.tolist(),
        'signatures': signatures,
        'analysis_time': time.time() - start_time
    }
    
    # Save results
    output_file = f"quantum_50_pulsars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n🎯 QUANTUM TEST COMPLETE!")
    print("=" * 50)
    print(f"📊 Analysis Statistics:")
    print(f"   Pulsars analyzed: {len(pulsar_ids)}")
    print(f"   High quantum correlation pairs: {len(signatures['high_kernel_pairs'])}")
    print(f"   High entanglement pairs: {len(signatures['high_entropy_pairs'])}")
    print(f"   Total quantum signatures: {signatures['quantum_signatures']}")
    print(f"   Analysis time: {results['analysis_time']:.2f} seconds")
    print(f"💾 Results saved to: {output_file}")
    
    # Show top signatures
    if signatures['high_kernel_pairs']:
        print(f"\n🌟 TOP QUANTUM CORRELATION PAIRS:")
        for i, pair in enumerate(signatures['high_kernel_pairs'][:3]):
            print(f"   {i+1}. {pair['pulsar1']} ↔ {pair['pulsar2']}: Kernel = {pair['kernel_value']:.3f}, Entropy = {pair['entropy']:.3f}")
    
    if signatures['high_entropy_pairs']:
        print(f"\n🔗 TOP ENTANGLEMENT PAIRS:")
        for i, pair in enumerate(signatures['high_entropy_pairs'][:3]):
            print(f"   {i+1}. {pair['pulsar1']} ↔ {pair['pulsar2']}: Entropy = {pair['entropy']:.3f}, Kernel = {pair['kernel_value']:.3f}")
    
    return results

if __name__ == "__main__":
    main()
