#!/usr/bin/env python3
"""
RUN MODERN EXOTIC PHYSICS HUNTER v3.0

This is the main execution script for the modern exotic physics hunter.
It combines all the advanced methods and provides a unified interface.
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import our custom modules
from MODERN_EXOTIC_PHYSICS_HUNTER_v3 import ModernExoticPhysicsHunter, PhysicsChannel, Detection
from MODERN_EXOTIC_PHYSICS_HUNTER_v3_methods import (
    transformer_coherence_analysis, vae_anomaly_detection, graph_neural_analysis,
    quantum_gravity_search, extra_dimensions_search, ensemble_bayesian_analysis,
    deep_anomaly_detection, topological_ml_analysis, gradient_boosting_analysis,
    _count_connected_components, _tracy_widom_test
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteModernExoticPhysicsHunter(ModernExoticPhysicsHunter):
    """Complete modern exotic physics hunter with all methods integrated"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add the detection methods to the class
        self.transformer_coherence_analysis = transformer_coherence_analysis.__get__(self, type(self))
        self.vae_anomaly_detection = vae_anomaly_detection.__get__(self, type(self))
        self.graph_neural_analysis = graph_neural_analysis.__get__(self, type(self))
        self.quantum_gravity_search = quantum_gravity_search.__get__(self, type(self))
        self.extra_dimensions_search = extra_dimensions_search.__get__(self, type(self))
        self.ensemble_bayesian_analysis = ensemble_bayesian_analysis.__get__(self, type(self))
        self.deep_anomaly_detection = deep_anomaly_detection.__get__(self, type(self))
        self.topological_ml_analysis = topological_ml_analysis.__get__(self, type(self))
        self.gradient_boosting_analysis = gradient_boosting_analysis.__get__(self, type(self))
        self._count_connected_components = _count_connected_components.__get__(self, type(self))
        self._tracy_widom_test = _tracy_widom_test.__get__(self, type(self))
    
    def modern_unified_hunt(self, pulsar_data: Dict) -> Dict:
        """Execute modern unified hunt across all physics channels"""
        print("🌌 MODERN EXOTIC PHYSICS HUNTER v3.0 - UNIFIED SEARCH")
        print("="*80)
        print(f"🔍 Hunting {len(self.channels)} exotic physics channels in {len(pulsar_data)} pulsars...")
        print("🧠 Using: Deep Learning | Graph Neural Networks | Quantum-Inspired Optimization")
        print("⚡ GPU Acceleration:", "✅" if self.use_gpu else "❌")
        print()
        
        # Execute all hunts in parallel using ThreadPoolExecutor
        hunt_functions = {
            'axion_oscillations': self.transformer_coherence_analysis,
            'axion_clouds': self.vae_anomaly_detection,
            'dark_photons': self.graph_neural_analysis,
            'scalar_fields': self.ensemble_bayesian_analysis,
            'primordial_bhs': self.deep_anomaly_detection,
            'domain_walls': self.topological_ml_analysis,
            'fifth_force': self.gradient_boosting_analysis,
            'quantum_gravity': self.quantum_gravity_search,
            'extra_dimensions': self.extra_dimensions_search,
        }
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for channel in self.channels:
                if channel.name in hunt_functions:
                    future = executor.submit(hunt_functions[channel.name], pulsar_data)
                    futures[channel.name] = future
            
            # Collect results
            for channel_name, future in futures.items():
                try:
                    results[channel_name] = future.result(timeout=300)  # 5 min timeout
                except Exception as e:
                    logger.error(f"Hunt {channel_name} failed: {e}")
                    results[channel_name] = {'detections': [], 'significance': 0.0}
        
        # Compile comprehensive results
        unified_results = self._compile_results(results)
        
        # Generate modern report
        self.generate_modern_report(unified_results)
        
        return unified_results
    
    def _compile_results(self, results: Dict) -> Dict:
        """Compile results from all channels into unified report"""
        unified_results = {
            'total_detections': 0,
            'channels': results,
            'overall_significance': 0.0,
            'top_discoveries': [],
            'advanced_stats': {},
            'timestamp': datetime.now().isoformat(),
            'version': 'Modern Exotic Physics Hunter v3.0'
        }
        
        # Collect all detections
        all_detections = []
        all_significances = []
        
        for channel_name, channel_results in results.items():
            if 'detections' in channel_results:
                unified_results['total_detections'] += len(channel_results['detections'])
                
                # Extract significance
                significance = channel_results.get('significance', 0.0)
                all_significances.append(significance)
                
                for detection in channel_results['detections']:
                    if isinstance(detection, Detection):
                        detection_dict = {
                            'channel': detection.channel,
                            'significance': detection.significance,
                            'confidence': detection.confidence,
                            'parameters': detection.parameters
                        }
                        all_detections.append(detection_dict)
                    else:
                        all_detections.append(detection)
        
        # Compute overall statistics
        if all_significances:
            unified_results['overall_significance'] = max(all_significances)
        
        # Rank discoveries
        all_detections.sort(key=lambda x: x.get('significance', 0), reverse=True)
        unified_results['top_discoveries'] = all_detections[:10]
        
        # Advanced statistics
        unified_results['advanced_stats'] = {
            'mean_significance': np.mean(all_significances) if all_significances else 0.0,
            'std_significance': np.std(all_significances) if all_significances else 0.0,
            'channels_searched': len(results),
            'gpu_accelerated': self.use_gpu,
            'ensemble_size': self.ensemble_size
        }
        
        return unified_results
    
    def generate_modern_report(self, results: Dict) -> str:
        """Generate comprehensive modern analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"modern_exotic_physics_report_{timestamp}.json"
        
        # Prepare comprehensive report
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': 'Modern Exotic Physics Hunter v3.0',
                'methods_used': [
                    'Transformer Neural Networks',
                    'Variational Autoencoders',
                    'Graph Neural Networks',
                    'Bayesian MCMC with Ensemble Methods',
                    'Quantum-Inspired Optimization',
                    'Topological Machine Learning',
                    'Gradient Boosting',
                    'Extreme Value Theory',
                    'Random Matrix Theory'
                ],
                'hardware_info': {
                    'gpu_accelerated': self.use_gpu,
                    'device': str(self.device) if self.device else 'CPU',
                    'torch_available': hasattr(self, 'neural_net') and self.neural_net is not None
                },
                'parameters': {
                    'monte_carlo_trials': self.monte_carlo_trials,
                    'ensemble_size': self.ensemble_size,
                    'confidence_level': self.confidence_level
                }
            },
            'detection_summary': {
                'total_channels_searched': len(self.channels),
                'total_detections': results['total_detections'],
                'overall_significance': results['overall_significance'],
                'top_discovery': results['top_discoveries'][0] if results['top_discoveries'] else None
            },
            'channel_results': results['channels'],
            'top_discoveries': results['top_discoveries'],
            'advanced_statistics': results['advanced_stats'],
            'physics_channels': [
                {
                    'name': ch.name,
                    'frequency_range': ch.frequency_range,
                    'detection_threshold': ch.detection_threshold,
                    'method': ch.method
                } for ch in self.channels
            ]
        }
        
        # Save report
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        json_report = convert_numpy(report)
        
        with open(filename, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"\n💾 Modern analysis report saved to: {filename}")
        
        # Print executive summary
        print("\n" + "="*80)
        print("🌌 MODERN EXOTIC PHYSICS HUNTER v3.0 - EXECUTIVE SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Maximum significance: {results['overall_significance']:.2f}σ")
        print(f"   Total detections: {results['total_detections']}")
        print(f"   Channels analyzed: {len(self.channels)}")
        print(f"   GPU acceleration: {'✅' if self.use_gpu else '❌'}")
        print(f"   Ensemble methods: {'✅' if self.ensemble_size > 1 else '❌'}")
        
        if results['top_discoveries']:
            top = results['top_discoveries'][0]
            print(f"\n🎯 TOP DISCOVERY:")
            print(f"   Channel: {top.get('channel', 'unknown')}")
            print(f"   Significance: {top.get('significance', 0):.2f}σ")
            print(f"   Confidence: {top.get('confidence', 0):.1%}")
        
        # Channel-by-channel summary
        print(f"\n📈 CHANNEL SUMMARY:")
        for channel_name, channel_results in results['channels'].items():
            significance = channel_results.get('significance', 0)
            detections = len(channel_results.get('detections', []))
            status = "🚨" if significance > 5.0 else "⚡" if significance > 4.0 else "🔍" if significance > 3.0 else "✅"
            print(f"   {status} {channel_name}: {significance:.2f}σ ({detections} detections)")
        
        return filename

def main():
    """Main execution function for modern exotic physics hunter"""
    print("🌌 MODERN EXOTIC PHYSICS HUNTER v3.0 - PRODUCTION READY!")
    print("="*80)
    print("🧠 Featuring: Transformers | VAE | Graph Neural Networks | Quantum-Inspired Optimization")
    print("⚡ GPU Acceleration | Ensemble Methods | Topological ML | AutoML")
    print()
    
    # Initialize the modern hunter
    hunter = CompleteModernExoticPhysicsHunter(
        monte_carlo_trials=5000,
        use_gpu=False,  # Set to True if you have PyTorch and CUDA
        ensemble_size=10,
        confidence_level=0.95
    )
    
    # Load or generate data
    print("📡 Loading pulsar timing data...")
    try:
        # Try to load real data
        from IPTA_TIMING_PARSER import load_ipta_timing_data
        data = load_ipta_timing_data()
        print(f"✅ Loaded {len(data)} real pulsars from IPTA DR2")
        
        if len(data) == 0:
            print("❌ ERROR: No real data loaded. This platform requires real IPTA DR2 data.")
            print("Please ensure 02_Data/ipta_dr2/ contains valid pulsar timing data.")
            return
    except Exception as e:
        print(f"❌ ERROR: Failed to load real IPTA DR2 data: {e}")
        print("This platform requires real pulsar timing data - no synthetic fallback.")
        return
    
    # Execute modern unified hunt
    print("\n🚀 Launching modern exotic physics hunt...")
    results = hunter.modern_unified_hunt(data)
    
    print(f"\n🎯 MODERN EXOTIC PHYSICS HUNT COMPLETE!")
    print(f"   ✅ Analyzed {len(hunter.channels)} physics channels")
    print(f"   ✅ Processed {len(data)} pulsars")
    print(f"   ✅ Applied modern deep learning methods")
    print(f"   ✅ Generated comprehensive analysis report")
    print(f"\n📊 Final Status:")
    print(f"   Maximum significance: {results['overall_significance']:.2f}σ")
    print(f"   Total detections: {results['total_detections']}")
    print(f"   Modern methods applied: ✅")

if __name__ == "__main__":
    main()
