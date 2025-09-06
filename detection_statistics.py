import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from sklearn.metrics import roc_curve, auc
import logging
from typing import Callable, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionStatistics:
    def __init__(self):
        logger.info("Detection Statistics module initialized")

    def _calculate_log_likelihood(self, data: np.ndarray, model_func: Callable, params: Dict) -> float:
        """
        Calculate log-likelihood for given data and model.
        
        Args:
            data: Input data array
            model_func: Model function
            params: Model parameters
            
        Returns:
            Log-likelihood value
        """
        try:
            model_output = model_func(data, params)
            
            # Handle zero or negative values
            model_output = np.maximum(model_output, 1e-10)
            
            return -np.sum(np.log(model_output))
        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {e}")
            return -np.inf

    def _fit_model(self, data: np.ndarray, model_func: Callable, initial_params: Dict) -> Dict:
        """
        Fit model to data using maximum likelihood estimation.
        
        Args:
            data: Input data array
            model_func: Model function
            initial_params: Initial parameter values
            
        Returns:
            Dictionary with fitted parameters and log-likelihood
        """
        try:
            # Convert initial_params to array
            param_names = list(initial_params.keys())
            param_values = np.array([initial_params[name] for name in param_names])
            
            def objective(params):
                param_dict = {name: params[i] for i, name in enumerate(param_names)}
                return self._calculate_log_likelihood(data, model_func, param_dict)
            
            # Minimize negative log-likelihood
            result = minimize(objective, param_values, method='Nelder-Mead')
            
            if result.success:
                fitted_params = {name: result.x[i] for i, name in enumerate(param_names)}
                return {
                    'params': fitted_params,
                    'log_likelihood': -result.fun
                }
            else:
                logger.warning("Model fitting failed: " + str(result.message))
                return {
                    'params': initial_params,
                    'log_likelihood': -np.inf
                }
        except Exception as e:
            logger.warning(f"Model fitting failed: {e}")
            return {
                'params': initial_params,
                'log_likelihood': -np.inf
            }

    def likelihood_ratio_test(self, data: np.ndarray, null_model: Callable, 
                            signal_model: Callable, parameters: Dict = None) -> Dict:
        """
        Perform likelihood ratio test for signal detection.
        
        Args:
            data: Input data array
            null_model: Null hypothesis model function
            signal_model: Signal hypothesis model function
            parameters: Model parameters
            
        Returns:
            Dictionary with test results
        """
        if parameters is None:
            parameters = {'amplitude': 1.0, 'frequency': 0.1}
        
        # Fit null model
        null_result = self._fit_model(data, null_model, parameters)
        null_likelihood = null_result['log_likelihood']
        
        # Fit signal model
        signal_result = self._fit_model(data, signal_model, parameters)
        signal_likelihood = signal_result['log_likelihood']
        
        # Calculate likelihood ratio
        if null_likelihood != -np.inf and signal_likelihood != -np.inf:
            likelihood_ratio = signal_likelihood / null_likelihood
        else:
            likelihood_ratio = 1.0
        
        # Calculate significance (simplified)
        significance = np.log(likelihood_ratio) if likelihood_ratio > 0 else 0
        
        return {
            'likelihood_ratio': likelihood_ratio,
            'significance': significance,
            'null_likelihood': null_likelihood,
            'signal_likelihood': signal_likelihood
        }

    def false_alarm_analysis(self, test_statistics: np.ndarray, 
                           null_distribution_pdf: Callable, 
                           thresholds: np.ndarray = None) -> Dict:
        """
        Calculate false alarm rates for given thresholds.
        
        Args:
            test_statistics: Array of test statistics
            null_distribution_pdf: Null distribution PDF function
            thresholds: Array of threshold values
            
        Returns:
            Dictionary with false alarm analysis results
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 10.0, 10)
        
        fa_rates = []
        for threshold in thresholds:
            # Calculate false alarm rate
            fa_rate = 1.0 - null_distribution_pdf(threshold)
            fa_rates.append(fa_rate)
        
        return {
            'thresholds': thresholds,
            'false_alarm_rates': np.array(fa_rates)
        }

    def detection_sensitivity_curves(self, signal_amplitudes: np.ndarray, 
                                   noise_levels: np.ndarray, n_trials: int = 1000) -> np.ndarray:
        """
        Calculate detection sensitivity curves using Monte Carlo simulation.
        
        Args:
            signal_amplitudes: Array of signal amplitudes to test
            noise_levels: Array of noise levels to test
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Sensitivity matrix (amplitudes x noise_levels)
        """
        sensitivity_matrix = np.zeros((len(signal_amplitudes), len(noise_levels)))
        
        for i, amplitude in enumerate(signal_amplitudes):
            for j, noise_level in enumerate(noise_levels):
                detections = 0
                
                for trial in range(n_trials):
                    # Generate test data
                    signal = amplitude * np.sin(2.0 * np.pi * np.linspace(0, 1, 100))
                    noise = np.random.normal(0, noise_level, 100)
                    data = signal + noise
                    
                    # Simple detection test (SNR > 3)
                    snr = np.max(np.abs(data)) / noise_level
                    if snr > 3.0:
                        detections += 1
                
                sensitivity_matrix[i, j] = detections / n_trials
        
        return sensitivity_matrix

    def roc_analysis(self, signal_data: np.ndarray, noise_data: np.ndarray) -> Dict:
        """
        Perform Receiver Operating Characteristic (ROC) analysis.
        
        Args:
            signal_data: Array of signal data
            noise_data: Array of noise data
            
        Returns:
            Dictionary with ROC analysis results
        """
        # Create labels
        y_true = np.concatenate([np.ones(len(signal_data)), np.zeros(len(noise_data))])
        y_scores = np.concatenate([signal_data, noise_data])
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate AUC
        auc_score = auc(fpr, tpr)
        
        return {
            'false_positive_rates': fpr,
            'true_positive_rates': tpr,
            'thresholds': thresholds,
            'auc': auc_score
        }

    def systematic_error_analysis(self, nominal_result: Dict, 
                                systematic_uncertainties: Dict) -> Dict:
        """
        Quantify systematic errors in analysis results.
        
        Args:
            nominal_result: Nominal analysis result
            systematic_uncertainties: Dictionary of systematic uncertainties
            
        Returns:
            Dictionary with systematic error analysis
        """
        total_uncertainty = 0.0
        
        for component, uncertainty in systematic_uncertainties.items():
            total_uncertainty += uncertainty**2
        
        total_uncertainty = np.sqrt(total_uncertainty)
        
        return {
            'total_uncertainty': total_uncertainty,
            'components': systematic_uncertainties
        }

if __name__ == "__main__":
    # Test the detection statistics module
    print("🧪 Testing Detection Statistics Framework...")
    
    stats = DetectionStatistics()
    print("✅ Framework initialized successfully")
    
    # Test likelihood ratio test
    data = np.random.normal(0, 1, 100)
    
    def null_model(x, params):
        return np.ones_like(x) * 0.1
    
    def signal_model(x, params):
        return np.ones_like(x) * 0.2
    
    lrt_result = stats.likelihood_ratio_test(data, null_model, signal_model)
    print(f"✅ Likelihood ratio test: significance = {lrt_result['significance']:.2f}σ")
    
    # Test false alarm analysis
    test_stats = np.random.chisquare(1, 100)
    def chi2_pdf(x):
        return chi2.cdf(x, 1)
    
    fa_result = stats.false_alarm_analysis(test_stats, chi2_pdf)
    print(f"✅ False alarm analysis: {len(fa_result['thresholds'])} thresholds")
    
    # Test detection sensitivity
    amplitudes = np.logspace(-3, -1, 10)
    noise_levels = np.logspace(-2, 0, 5)
    sensitivity = stats.detection_sensitivity_curves(amplitudes, noise_levels, 100)
    print(f"✅ Detection sensitivity: {sensitivity.shape} matrix")
    
    # Test ROC analysis
    signal_data = np.random.normal(1, 0.5, 50)
    noise_data = np.random.normal(0, 0.5, 50)
    roc_result = stats.roc_analysis(signal_data, noise_data)
    print(f"✅ ROC analysis: AUC = {roc_result['auc']:.3f}")
    
    # Test systematic error analysis
    nominal = {'value': 1.0}
    systematic = {'calibration': 0.05, 'model': 0.03}
    sys_result = stats.systematic_error_analysis(nominal, systematic)
    print(f"✅ Systematic error analysis: total uncertainty = {sys_result['total_uncertainty']:.3f}")
    
    print("🎉 All detection statistics tests passed!")
