"""
Frequency-domain feature extraction from sensor signals.

Extracts 26 features from accelerometer, HR, and PPG signals using
Welch's method for PSD estimation, including:
- Per accelerometer axis (Ax, Ay, Az): dominant freq, power, spectral entropy,
  band power [0.5-5Hz], [5-15Hz], [15-50Hz], spectral centroid, spectral spread
- HR signal: dominant freq, spectral entropy, LF power, HF power
- PPG signal: dominant freq, spectral purity, SNR
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal as scipy_signal

from .base import BaseFeatureExtractor


class FrequencyDomainFeatures(BaseFeatureExtractor):
    """Extract frequency-domain features from sensor signals.
    
    This extractor computes spectral features using Welch's method for
    power spectral density estimation across accelerometer, HR, and
    PPG signals.
    
    Features (26 total):
        For each accelerometer axis (Ax, Ay, Az) - 8 features each = 24:
            freq_ax_dominant_freq, freq_ax_dominant_power, freq_ax_spectral_entropy,
            freq_ax_power_0_5_5, freq_ax_power_5_15, freq_ax_power_15_50,
            freq_ax_spectral_centroid, freq_ax_spectral_spread
            (same for Ay, Az)
        
        HR signal - 4 features:
            freq_hr_dominant_freq, freq_hr_spectral_entropy,
            freq_hr_lf_power, freq_hr_hf_power
        
        PPG signal - 3 features (minus 1 if not available):
            freq_ppg_dominant_freq, freq_ppg_spectral_purity, freq_ppg_snr
    
    Note:
        Uses Welch's method with configurable parameters for PSD estimation.
        Default sampling rate: 30 Hz (adjustable via metadata).
    """
    
    # Default sampling rates
    DEFAULT_ACCEL_FS: float = 30.0
    DEFAULT_HR_FS: float = 4.0  # HR typically sampled at lower rate
    DEFAULT_PPG_FS: float = 100.0
    
    def __init__(self) -> None:
        """Initialize frequency domain feature extractor."""
        super().__init__()
        self.feature_names = [
            # Accelerometer Ax features
            "freq_ax_dominant_freq", "freq_ax_dominant_power",
            "freq_ax_spectral_entropy", "freq_ax_power_0_5_5",
            "freq_ax_power_5_15", "freq_ax_power_15_50",
            "freq_ax_spectral_centroid", "freq_ax_spectral_spread",
            # Accelerometer Ay features
            "freq_ay_dominant_freq", "freq_ay_dominant_power",
            "freq_ay_spectral_entropy", "freq_ay_power_0_5_5",
            "freq_ay_power_5_15", "freq_ay_power_15_50",
            "freq_ay_spectral_centroid", "freq_ay_spectral_spread",
            # Accelerometer Az features
            "freq_az_dominant_freq", "freq_az_dominant_power",
            "freq_az_spectral_entropy", "freq_az_power_0_5_5",
            "freq_az_power_5_15", "freq_az_power_15_50",
            "freq_az_spectral_centroid", "freq_az_spectral_spread",
            # HR features
            "freq_hr_dominant_freq", "freq_hr_spectral_entropy",
            "freq_hr_lf_power", "freq_hr_hf_power",
            # PPG features
            "freq_ppg_dominant_freq", "freq_ppg_spectral_purity",
            "freq_ppg_snr",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract frequency-domain features from sensor signals.
        
        Args:
            window_data: Dictionary containing sensor data with keys
                'accelerometer', 'heart_rate', and optionally 'ppg_raw'.
        
        Returns:
            Dictionary of frequency-domain feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Get sensor data
        ax = self._get_sensor_array(window_data, "accelerometer", "ax")
        ay = self._get_sensor_array(window_data, "accelerometer", "ay")
        az = self._get_sensor_array(window_data, "accelerometer", "az")
        hr = self._get_sensor_array(window_data, "heart_rate", "bpm")
        ppg = self._get_sensor_array(window_data, "heart_rate", "ppg_raw")
        
        # Get sampling rates from metadata
        accel_fs = self._get_nested(
            window_data, "metadata", "sampling_rate", default=self.DEFAULT_ACCEL_FS
        )
        hr_fs = self._get_nested(
            window_data, "metadata", "hr_sampling_rate", default=self.DEFAULT_HR_FS
        )
        ppg_fs = self._get_nested(
            window_data, "metadata", "ppg_sampling_rate", default=self.DEFAULT_PPG_FS
        )
        
        # Extract accelerometer features for each axis
        for axis_name, axis_data in [("ax", ax), ("ay", ay), ("az", az)]:
            features.update(self._extract_accel_axis_features(
                axis_data, f"freq_{axis_name}", accel_fs
            ))
        
        # Extract HR features
        features.update(self._extract_hr_features(hr, hr_fs))
        
        # Extract PPG features
        features.update(self._extract_ppg_features(ppg, ppg_fs))
        
        return features
    
    def _extract_accel_axis_features(
        self, arr: np.ndarray, prefix: str, fs: float
    ) -> Dict[str, float]:
        """Extract frequency features for one accelerometer axis.
        
        Args:
            arr: 1D array of accelerometer data.
            prefix: Feature name prefix (e.g., 'freq_ax').
            fs: Sampling frequency in Hz.
        
        Returns:
            Dictionary of features for this axis.
        """
        features: Dict[str, float] = {}
        
        if len(arr) < 10:
            # Return default values
            features[f"{prefix}_dominant_freq"] = 0.0
            features[f"{prefix}_dominant_power"] = 0.0
            features[f"{prefix}_spectral_entropy"] = 0.0
            features[f"{prefix}_power_0_5_5"] = 0.0
            features[f"{prefix}_power_5_15"] = 0.0
            features[f"{prefix}_power_15_50"] = 0.0
            features[f"{prefix}_spectral_centroid"] = 0.0
            features[f"{prefix}_spectral_spread"] = 0.0
            return features
        
        def _compute_freq_features(x: np.ndarray) -> Dict[str, float]:
            result: Dict[str, float] = {}
            
            # Compute PSD using Welch's method
            nperseg = min(len(x), 128)
            if nperseg < 16:
                nperseg = len(x)
            
            try:
                freqs, psd = scipy_signal.welch(
                    x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
                )
            except Exception:
                return {k: 0.0 for k in [
                    f"{prefix}_dominant_freq", f"{prefix}_dominant_power",
                    f"{prefix}_spectral_entropy", f"{prefix}_power_0_5_5",
                    f"{prefix}_power_5_15", f"{prefix}_power_15_50",
                    f"{prefix}_spectral_centroid", f"{prefix}_spectral_spread",
                ]}
            
            if len(freqs) == 0 or np.sum(psd) == 0:
                return {k: 0.0 for k in [
                    f"{prefix}_dominant_freq", f"{prefix}_dominant_power",
                    f"{prefix}_spectral_entropy", f"{prefix}_power_0_5_5",
                    f"{prefix}_power_5_15", f"{prefix}_power_15_50",
                    f"{prefix}_spectral_centroid", f"{prefix}_spectral_spread",
                ]}
            
            # Dominant frequency and its power
            peak_idx = np.argmax(psd)
            result[f"{prefix}_dominant_freq"] = float(freqs[peak_idx])
            result[f"{prefix}_dominant_power"] = float(psd[peak_idx])
            
            # Spectral entropy
            result[f"{prefix}_spectral_entropy"] = float(
                self._spectral_entropy(freqs, psd)
            )
            
            # Band powers
            result[f"{prefix}_power_0_5_5"] = float(self._band_power(freqs, psd, 0.5, 5.0))
            result[f"{prefix}_power_5_15"] = float(self._band_power(freqs, psd, 5.0, 15.0))
            result[f"{prefix}_power_15_50"] = float(self._band_power(freqs, psd, 15.0, 50.0))
            
            # Spectral centroid
            result[f"{prefix}_spectral_centroid"] = float(
                self._spectral_centroid(freqs, psd)
            )
            
            # Spectral spread
            result[f"{prefix}_spectral_spread"] = float(
                self._spectral_spread(freqs, psd)
            )
            
            return result
        
        features = self._safe_extract(arr, _compute_freq_features, default={
            k: 0.0 for k in [
                f"{prefix}_dominant_freq", f"{prefix}_dominant_power",
                f"{prefix}_spectral_entropy", f"{prefix}_power_0_5_5",
                f"{prefix}_power_5_15", f"{prefix}_power_15_50",
                f"{prefix}_spectral_centroid", f"{prefix}_spectral_spread",
            ]
        })
        return features
    
    def _extract_hr_features(self, hr: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract frequency features from heart rate signal.
        
        Args:
            hr: Heart rate BPM array.
            fs: Sampling frequency of HR signal.
        
        Returns:
            Dictionary of HR frequency features.
        """
        features: Dict[str, float] = {}
        
        if len(hr) < 10:
            features["freq_hr_dominant_freq"] = 0.0
            features["freq_hr_spectral_entropy"] = 0.0
            features["freq_hr_lf_power"] = 0.0
            features["freq_hr_hf_power"] = 0.0
            return features
        
        def _compute_hr_freq(x: np.ndarray) -> Dict[str, float]:
            result: Dict[str, float] = {}
            
            # Compute PSD
            nperseg = min(len(x), 64)
            if nperseg < 16:
                nperseg = len(x)
            
            try:
                freqs, psd = scipy_signal.welch(
                    x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
                )
            except Exception:
                return {k: 0.0 for k in [
                    "freq_hr_dominant_freq", "freq_hr_spectral_entropy",
                    "freq_hr_lf_power", "freq_hr_hf_power",
                ]}
            
            if len(freqs) == 0 or np.sum(psd) == 0:
                return {k: 0.0 for k in [
                    "freq_hr_dominant_freq", "freq_hr_spectral_entropy",
                    "freq_hr_lf_power", "freq_hr_hf_power",
                ]}
            
            # Dominant frequency
            peak_idx = np.argmax(psd)
            result["freq_hr_dominant_freq"] = float(freqs[peak_idx])
            
            # Spectral entropy
            result["freq_hr_spectral_entropy"] = float(
                self._spectral_entropy(freqs, psd)
            )
            
            # LF and HF power
            result["freq_hr_lf_power"] = float(
                self._band_power(freqs, psd, 0.04, 0.15)
            )
            result["freq_hr_hf_power"] = float(
                self._band_power(freqs, psd, 0.15, 0.4)
            )
            
            return result
        
        features = self._safe_extract(hr, _compute_hr_freq, default={
            k: 0.0 for k in [
                "freq_hr_dominant_freq", "freq_hr_spectral_entropy",
                "freq_hr_lf_power", "freq_hr_hf_power",
            ]
        })
        return features
    
    def _extract_ppg_features(self, ppg: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract frequency features from PPG signal.
        
        Args:
            ppg: Raw PPG waveform array.
            fs: Sampling frequency of PPG signal.
        
        Returns:
            Dictionary of PPG frequency features.
        """
        features: Dict[str, float] = {}
        
        if len(ppg) < 10:
            features["freq_ppg_dominant_freq"] = 0.0
            features["freq_ppg_spectral_purity"] = 0.0
            features["freq_ppg_snr"] = 0.0
            return features
        
        def _compute_ppg_freq(x: np.ndarray) -> Dict[str, float]:
            result: Dict[str, float] = {}
            
            # Compute PSD
            nperseg = min(len(x), 128)
            if nperseg < 16:
                nperseg = len(x)
            
            try:
                freqs, psd = scipy_signal.welch(
                    x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
                )
            except Exception:
                return {k: 0.0 for k in [
                    "freq_ppg_dominant_freq", "freq_ppg_spectral_purity",
                    "freq_ppg_snr",
                ]}
            
            if len(freqs) == 0 or np.sum(psd) == 0:
                return {k: 0.0 for k in [
                    "freq_ppg_dominant_freq", "freq_ppg_spectral_purity",
                    "freq_ppg_snr",
                ]}
            
            # Dominant frequency (should match HR)
            peak_idx = np.argmax(psd)
            result["freq_ppg_dominant_freq"] = float(freqs[peak_idx])
            
            # Spectral purity: ratio of peak power to total power
            total_power = np.sum(psd)
            if total_power > 0:
                result["freq_ppg_spectral_purity"] = float(psd[peak_idx] / total_power)
            else:
                result["freq_ppg_spectral_purity"] = 0.0
            
            # SNR: signal power in 0.5-4 Hz band vs noise
            signal_band = (freqs >= 0.5) & (freqs <= 4.0)
            noise_band = (freqs > 4.0) | (freqs < 0.3)
            
            signal_power = np.sum(psd[signal_band]) if np.any(signal_band) else 0.0
            noise_power = np.sum(psd[noise_band]) if np.any(noise_band) else 1.0
            
            if noise_power > 0:
                result["freq_ppg_snr"] = float(10 * np.log10(signal_power / noise_power))
            else:
                result["freq_ppg_snr"] = 0.0
            
            return result
        
        features = self._safe_extract(ppg, _compute_ppg_freq, default={
            k: 0.0 for k in [
                "freq_ppg_dominant_freq", "freq_ppg_spectral_purity",
                "freq_ppg_snr",
            ]
        })
        return features
    
    def _spectral_entropy(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Compute spectral entropy of PSD.
        
        Args:
            freqs: Frequency array.
            psd: Power spectral density array.
        
        Returns:
            Spectral entropy value (nats).
        """
        if len(psd) == 0 or np.sum(psd) == 0:
            return 0.0
        
        # Normalize PSD to probability distribution
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        
        if len(psd_norm) == 0:
            return 0.0
        
        # Shannon entropy
        entropy = -np.sum(psd_norm * np.log(psd_norm))
        return float(entropy)
    
    def _band_power(
        self, freqs: np.ndarray, psd: np.ndarray, low: float, high: float
    ) -> float:
        """Compute power in a frequency band using trapezoidal integration.
        
        Args:
            freqs: Frequency array.
            psd: Power spectral density array.
            low: Lower frequency bound.
            high: Upper frequency bound.
        
        Returns:
            Power in the specified band.
        """
        mask = (freqs >= low) & (freqs <= high)
        if np.sum(mask) < 2:
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))
    
    def _spectral_centroid(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Compute spectral centroid (center of mass of spectrum).
        
        Args:
            freqs: Frequency array.
            psd: Power spectral density array.
        
        Returns:
            Spectral centroid in Hz.
        """
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0
        return float(np.sum(freqs * psd) / total_power)
    
    def _spectral_spread(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Compute spectral spread (standard deviation of spectrum).
        
        Args:
            freqs: Frequency array.
            psd: Power spectral density array.
        
        Returns:
            Spectral spread in Hz.
        """
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0
        
        centroid = np.sum(freqs * psd) / total_power
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power)
        return float(spread)