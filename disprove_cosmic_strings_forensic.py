#!/usr/bin/env python3
"""
DISPROVE COSMIC STRINGS – FORENSIC EDITION
==========================================
Try to kill the detection.  If you can't, the signal survives.
Auto-flags synthetic data artifacts.
"""

import json
import sys
import numpy as np
import logging
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("forensic")


# ------------------------------------------------------------------
# Hard-coded physics constants
# ------------------------------------------------------------------
EXPECTED_HD_AMP = 1e-15          # nominal GWB amplitude
SLOPE_TOL = 0.1                  # |slope| < 0.1  →  white-spectrum
Gμ2A = 1.6e-14                   # Gμ=1e-11  →  A≈1.6e-14 (Sesana et al. 2021)
# ------------------------------------------------------------------


class ForensicDisprover:
    """3× disproof + toy-data autopsy"""

    def __init__(self, json_path: str):
        self.js: Dict[str, Any] = json.load(open(json_path))
        self.report: Dict[str, Any] = {"toy_red_flags": [], "disproof": []}

    # ------------------------------------------------------------------
    # Red-flag detectors
    # ------------------------------------------------------------------
    def _flag_toy_data(self) -> None:
        # Check for toy data red flags based on our actual data structure
        pa = self.js["periodic_analysis"]
        ca = self.js["correlation_analysis"]
        
        # Check for perfect detection rates (toy data signature)
        if pa["detection_rate"] == 100.0:
            self.report["toy_red_flags"].append("PERFECT_DETECTION_RATE")
        if ca["detection_rate"] == 100.0:
            self.report["toy_red_flags"].append("PERFECT_CORRELATION_DETECTION")
            
        # Check for unrealistic FAP values
        if "mean_fap" in pa and pa["mean_fap"] == 0.0:
            self.report["toy_red_flags"].append("ZERO_FAP_EVERYWHERE")
            
        # Check for uniform correlations (toy data signature)
        if "mean_correlation" in ca and "std_correlation" in ca:
            if ca["std_correlation"] < 0.01:  # Very low variance in correlations
                self.report["toy_red_flags"].append("UNIFORM_CORRELATIONS")

    # ------------------------------------------------------------------
    # 1.  Correlation disproof  (proper HD χ²)
    # ------------------------------------------------------------------
    def _disprove_correlations(self) -> str:
        ca = self.js["correlation_analysis"]
        
        # Use our actual data structure
        if "correlations" in ca and "angular_separations" in ca:
            xi_obs = ca["correlations"]  # list of rho_ij
            gamma_ij = ca["angular_separations"]  # rad
            gamma_ij = np.array(gamma_ij)
            xi_hd = 0.5 * (1 - np.cos(gamma_ij)) * np.log(0.5 * (1 - np.cos(gamma_ij))) - \
                   0.25 * (1 - np.cos(gamma_ij)) + 0.25 * (3 + np.cos(gamma_ij)) * np.log(0.5 * (3 + np.cos(gamma_ij)))

            chi2 = np.sum((np.array(xi_obs) - EXPECTED_HD_AMP * xi_hd) ** 2) / (len(xi_obs) - 1)
            fail = chi2 < 1.0  # crude threshold
        else:
            # Fallback to simpler test
            mean_corr = ca.get("mean_correlation", 0)
            fail = abs(mean_corr) > 0.1  # Strong correlation suggests signal
            chi2 = mean_corr
            
        self.report["disproof"].append(
            {"test": "HD_correlations", "chi2": chi2, "disproof": "FAILED" if fail else "SUCCESS"}
        )
        return "FAILED" if fail else "SUCCESS"

    # ------------------------------------------------------------------
    # 2.  Spectral disproof  →  95 % UL on Gμ
    # ------------------------------------------------------------------
    def _disprove_spectral(self) -> str:
        sa = self.js["spectral_analysis"]
        
        # Use our actual data structure
        slope = sa.get("mean_slope", 0)
        white = sa.get("mean_white_noise_strength", 1e-15)
        
        # 95 % UL on amplitude assuming power-law model
        A_ul = white * 10 ** (1.96 * slope)  # approximate
        Gμ_ul = A_ul / Gμ2A
        self.report["Gμ_95_upper_limit"] = Gμ_ul
        
        # Spectral disproof: if slope is too flat and white noise is too high
        fail = abs(slope) < SLOPE_TOL and white > 0.1 * EXPECTED_HD_AMP
        
        self.report["disproof"].append(
            {"test": "spectral_shape", "slope": slope, "Gμ_ul": Gμ_ul, "disproof": "FAILED" if fail else "SUCCESS"}
        )
        return "FAILED" if fail else "SUCCESS"

    # ------------------------------------------------------------------
    # 3.  Periodic disproof
    # ------------------------------------------------------------------
    def _disprove_periodic(self) -> str:
        pa = self.js["periodic_analysis"]
        
        # Use our actual data structure
        mean_fap = pa.get("mean_fap", 0.5)
        mean_snr = pa.get("mean_snr", 1.0)
        detection_rate = pa.get("detection_rate", 0)
        
        # Require < 1 % false alarms AND high detection rate
        fail = mean_fap < 0.01 and detection_rate > 50.0
        
        self.report["disproof"].append(
            {"test": "periodic_signals", "mean_fap": mean_fap, "mean_snr": mean_snr, "detection_rate": detection_rate,
             "disproof": "FAILED" if fail else "SUCCESS"}
        )
        return "FAILED" if fail else "SUCCESS"

    # ------------------------------------------------------------------
    # Run everything
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._flag_toy_data()
        d1 = self._disprove_correlations()
        d2 = self._disprove_spectral()
        d3 = self._disprove_periodic()

        fails = sum("FAILED" in d["disproof"] for d in self.report["disproof"])
        self.report["summary"] = {
            "total_disproof_tests": 3,
            "disproof_failures": fails,  # strong signal
            "disproof_successes": 3 - fails,
            "verdict": "TOY_DATA" if self.report["toy_red_flags"] else ("STRONG" if fails > 1 else "WEAK"),
        }

        with open("DISPROVE_FORENSIC_REPORT.json", "w") as fp:
            json.dump(self.report, fp, indent=2)

        # One-line stdout for pipelines
        print(self.report["summary"]["verdict"])


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:  disprove_cosmic_strings_forensic.py  <results.json>")
        sys.exit(1)
    ForensicDisprover(sys.argv[1]).run()
