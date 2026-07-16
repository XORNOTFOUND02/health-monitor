"""
Cross-sensor feature extraction combining multiple data streams.

Extracts 15 features that combine information from multiple sensors:
- HR x Temp coupling
- HR x SpO2 ratio
- Activity-adjusted HR
- Cardio-respiratory index
- Fatigue index
- Autonomic balance
- Motion-corrected HR
- Temperature-HRV correlation
- SpO2-HR coherence
- Sleep quality composite
- Combined fever score
- Fall risk score
- Dehydration proxy
- Cardiovascular strain
- Recovery index
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from .base import BaseFeatureExtractor


class CrossSensorFeatures(BaseFeatureExtractor):
    """Extract features that combine data from multiple sensors.
    
    This extractor computes derived features that require data from
    multiple sensor modalities within each analysis window.
    
    Features (15 total):
        xsensor_hr_temp_correlation: Correlation between HR and temperature
        xsensor_hr_spo2_ratio: mean_hr / mean_spo2
        xsensor_activity_adjusted_hr: (mean_hr - resting_hr) / movement_intensity
        xsensor_cardio_respiratory_index: (hr_mean * spo2_mean) / 100
        xsensor_fatigue_index: Complex fatigue metric
        xsensor_autonomic_balance: LF/HF ratio x HRV
        xsensor_motion_corrected_hr: HR filtered for motion artifacts
        xsensor_temp_hrv_correlation: Temperature-HRV correlation
        xsensor_spo2_hr_coherence: Cross-spectral density at resp frequency
        xsensor_sleep_quality: Sleep quality composite score
        xsensor_fever_score: Combined fever indicator
        xsensor_fall_risk: Fall risk composite score
        xsensor_dehydration_proxy: Dehydration risk metric
        xsensor_cardiovascular_strain: Cardiovascular strain metric
        xsensor_recovery_index: Rate of HR decrease after activity
    
    Note:
        Many features use approximate values when exact data is unavailable.
        Default assumptions:
        - Resting HR: 60 BPM
        - Window sampling rate: 30 Hz (adjustable)
    """
    
    # Default constants
    RESTING_HR_DEFAULT: float = 60.0
    WINDOW_DURATION_SEC: float = 30.0
    SAMPLING_RATE_DEFAULT: float = 30.0
    
    def __init__(self) -> None:
        """Initialize cross-sensor feature extractor."""
        super().__init__()
        self.feature_names = [
            "xsensor_hr_temp_correlation", "xsensor_hr_spo2_ratio",
            "xsensor_activity_adjusted_hr", "xsensor_cardio_respiratory_index",
            "xsensor_fatigue_index", "xsensor_autonomic_balance",
            "xsensor_motion_corrected_hr", "xsensor_temp_hrv_correlation",
            "xsensor_spo2_hr_coherence", "xsensor_sleep_quality",
            "xsensor_fever_score", "xsensor_fall_risk",
            "xsensor_dehydration_proxy", "xsensor_cardiovascular_strain",
            "xsensor_recovery_index",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract cross-sensor features from combined sensor data.
        
        Args:
            window_data: Dictionary containing all sensor data.
        
        Returns:
            Dictionary of cross-sensor feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Extract sensor data
        hr = self._get_sensor_array(window_data, "heart_rate", "bpm")
        spo2 = self._get_sensor_array(window_data, "heart_rate", "spo2")
        stts22h = self._get_sensor_array(window_data, "temperature", "stts22h_celsius")
        lm35 = self._get_sensor_array(window_data, "temperature", "lm35_celsius")
        ax = self._get_sensor_array(window_data, "accelerometer", "ax")
        ay = self._get_sensor_array(window_data, "accelerometer", "ay")
        az = self._get_sensor_array(window_data, "accelerometer", "az")
        
        # Get sampling rate from metadata
        fs = self._get_nested(window_data, "metadata", "sampling_rate", default=self.SAMPLING_RATE_DEFAULT)
        
        # Compute derived signals
        temp = self._get_temperature(stts22h, lm35)
        movement_intensity = self._compute_movement_intensity(ax, ay, az)
        hr_mean = np.mean(hr) if len(hr) > 0 else self.RESTING_HR_DEFAULT
        spo2_mean = np.mean(spo2) if len(spo2) > 0 else 98.0
        temp_mean = np.mean(temp) if len(temp) > 0 else 36.8
        
        # HR x Temp correlation
        features["xsensor_hr_temp_correlation"] = self._compute_signal_correlation(hr, temp)
        
        # HR x SpO2 ratio
        if spo2_mean > 0:
            features["xsensor_hr_spo2_ratio"] = hr_mean / spo2_mean
        else:
            features["xsensor_hr_spo2_ratio"] = 0.0
        
        # Activity-adjusted HR
        features["xsensor_activity_adjusted_hr"] = self._compute_activity_adjusted_hr(
            hr, movement_intensity
        )
        
        # Cardio-respiratory index
        features["xsensor_cardio_respiratory_index"] = (hr_mean * spo2_mean) / 100.0
        
        # Fatigue index
        features["xsensor_fatigue_index"] = self._compute_fatigue_index(
            hr, movement_intensity
        )
        
        # Autonomic balance
        features["xsensor_autonomic_balance"] = self._compute_autonomic_balance(hr)
        
        # Motion-corrected HR
        features["xsensor_motion_corrected_hr"] = self._compute_motion_corrected_hr(
            hr, ax, ay, az
        )
        
        # Temperature-HRV correlation
        features["xsensor_temp_hrv_correlation"] = self._compute_temp_hrv_correlation(
            hr, temp
        )
        
        # SpO2-HR coherence
        features["xsensor_spo2_hr_coherence"] = self._compute_spo2_hr_coherence(
            spo2, hr, fs
        )
        
        # Sleep quality composite
        features["xsensor_sleep_quality"] = self._compute_sleep_quality(
            ax, ay, az, hr, spo2
        )
        
        # Combined fever score
        features["xsensor_fever_score"] = self._compute_fever_score(temp, hr)
        
        # Fall risk score
        features["xsensor_fall_risk"] = self._compute_fall_risk(
            ax, ay, az, hr
        )
        
        # Dehydration proxy
        features["xsensor_dehydration_proxy"] = self._compute_dehydration_proxy(
            temp, hr, movement_intensity
        )
        
        # Cardiovascular strain
        features["xsensor_cardiovascular_strain"] = self._compute_cardiovascular_strain(
            hr, spo2
        )
        
        # Recovery index
        features["xsensor_recovery_index"] = self._compute_recovery_index(hr, ax, ay, az)
        
        return features
    
    def _get_temperature(
        self, stts22h: np.ndarray, lm35: np.ndarray
    ) -> np.ndarray:
        """Get combined temperature data, preferring STTS22H.
        
        Args:
            stts22h: STTS22H temperature array.
            lm35: LM35 temperature array.
        
        Returns:
            Temperature array.
        """
        if len(stts22h) > 0:
            return stts22h
        elif len(lm35) > 0:
            return lm35
        return np.array([], dtype=np.float64)
    
    def _compute_movement_intensity(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> float:
        """Compute movement intensity from accelerometer data.
        
        Args:
            ax: X-axis acceleration.
            ay: Y-axis acceleration.
            az: Z-axis acceleration.
        
        Returns:
            Movement intensity (std of magnitude).
        """
        min_len = min(len(ax), len(ay), len(az))
        if min_len < 2:
            return 0.0
        
        mag = np.sqrt(ax[:min_len] ** 2 + ay[:min_len] ** 2 + az[:min_len] ** 2)
        return float(np.std(mag, ddof=1))
    
    def _compute_signal_correlation(
        self, sig1: np.ndarray, sig2: np.ndarray
    ) -> float:
        """Compute correlation between two signals, resampled to common length.
        
        Args:
            sig1: First signal.
            sig2: Second signal.
        
        Returns:
            Pearson correlation coefficient.
        """
        min_len = min(len(sig1), len(sig2))
        if min_len < 3:
            return 0.0
        
        s1 = sig1[:min_len]
        s2 = sig2[:min_len]
        
        # Check for constant signals
        if np.std(s1) == 0 or np.std(s2) == 0:
            return 0.0
        
        corr = np.corrcoef(s1, s2)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)
    
    def _compute_activity_adjusted_hr(
        self, hr: np.ndarray, movement_intensity: float
    ) -> float:
        """Compute activity-adjusted HR: (mean_hr - resting_hr) / movement_intensity.
        
        Args:
            hr: Heart rate array.
            movement_intensity: Movement intensity value.
        
        Returns:
            Activity-adjusted HR ratio.
        """
        if len(hr) == 0:
            return 0.0
        
        mean_hr = np.mean(hr)
        if movement_intensity < 0.01:
            return 0.0
        
        return float((mean_hr - self.RESTING_HR_DEFAULT) / movement_intensity)
    
    def _compute_fatigue_index(
        self, hr: np.ndarray, movement_intensity: float
    ) -> float:
        """Compute fatigue index.
        
        Fatigue index = (1 / (rmssd / 20)) * (resting_hr_elevation / 5) * (1 / movement_intensity)
        
        Higher values indicate greater fatigue.
        
        Args:
            hr: Heart rate array.
            movement_intensity: Movement intensity value.
        
        Returns:
            Fatigue index value.
        """
        if len(hr) < 2:
            return 0.0
        
        # Compute RMSSD from HR (approximation)
        diffs = np.diff(hr)
        rmssd = np.sqrt(np.mean(diffs ** 2))
        
        if rmssd < 0.01:
            rmssd = 0.01
        
        # Resting HR elevation
        hr_elevation = np.mean(hr) - self.RESTING_HR_DEFAULT
        if hr_elevation < 0:
            hr_elevation = 0.0
        
        # Movement adjustment
        if movement_intensity < 0.01:
            movement_intensity = 0.01
        
        fatigue = (1.0 / (rmssd / 20.0)) * (hr_elevation / 5.0) * (1.0 / movement_intensity)
        
        # Clamp to reasonable range
        return float(np.clip(fatigue, 0.0, 10.0))
    
    def _compute_autonomic_balance(self, hr: np.ndarray) -> float:
        """Compute autonomic balance proxy using HRV estimate.
        
        Uses HR variability as a proxy for autonomic balance.
        Higher HRV generally indicates better autonomic balance.
        
        Args:
            hr: Heart rate array.
        
        Returns:
            Autonomic balance score (0-1).
        """
        if len(hr) < 3:
            return 0.5  # Default
        
        # Compute HRV proxy
        hr_std = np.std(hr, ddof=1)
        
        # Normalize: typical HRV std is 2-10 BPM
        balance = min(1.0, hr_std / 10.0)
        return float(balance)
    
    def _compute_motion_corrected_hr(
        self, hr: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> float:
        """Compute motion-corrected HR by filtering out motion artifact periods.
        
        Args:
            hr: Heart rate array.
            ax, ay, az: Accelerometer arrays.
        
        Returns:
            Corrected mean HR value.
        """
        if len(hr) == 0:
            return self.RESTING_HR_DEFAULT
        
        min_len_acc = min(len(ax), len(ay), len(az))
        
        if min_len_acc == 0 or min_len_acc != len(hr):
            return float(np.mean(hr))
        
        # Compute movement magnitude
        mag = np.sqrt(ax[:min_len_acc] ** 2 + ay[:min_len_acc] ** 2 + az[:min_len_acc] ** 2)
        
        # Identify low-motion periods
        motion_threshold = np.mean(mag) + np.std(mag)
        low_motion_mask = mag < motion_threshold
        
        if np.sum(low_motion_mask) < 3:
            return float(np.mean(hr))
        
        # Return HR during low-motion periods
        return float(np.mean(hr[low_motion_mask]))
    
    def _compute_temp_hrv_correlation(
        self, hr: np.ndarray, temp: np.ndarray
    ) -> float:
        """Compute correlation between temperature and HR variability.
        
        Args:
            hr: Heart rate array.
            temp: Temperature array.
        
        Returns:
            Correlation coefficient.
        """
        if len(hr) < 3 or len(temp) < 3:
            return 0.0
        
        # Resample to common length
        min_len = min(len(hr), len(temp))
        hr_resampled = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(hr)),
            hr,
        )
        temp_resampled = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(temp)),
            temp,
        )
        
        return self._compute_signal_correlation(hr_resampled, temp_resampled)
    
    def _compute_spo2_hr_coherence(
        self, spo2: np.ndarray, hr: np.ndarray, fs: float
    ) -> float:
        """Compute SpO2-HR coherence at respiratory frequency.
        
        Respiratory frequency typically 0.15-0.4 Hz.
        
        Args:
            spo2: SpO2 array.
            hr: Heart rate array.
            fs: Sampling frequency.
        
        Returns:
            Coherence value (0-1).
        """
        min_len = min(len(spo2), len(hr))
        if min_len < 20:
            return 0.0
        
        spo2_r = spo2[:min_len]
        hr_r = hr[:min_len]
        
        try:
            # Compute coherence
            freqs, coh = scipy_signal.coherence(
                spo2_r, hr_r, fs=fs, nperseg=min(64, min_len)
            )
            
            # Get coherence in respiratory band (0.15-0.4 Hz)
            resp_mask = (freqs >= 0.15) & (freqs <= 0.4)
            if np.sum(resp_mask) > 0:
                return float(np.max(coh[resp_mask]))
            return 0.0
        except Exception:
            return 0.0
    
    def _compute_sleep_quality(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
        hr: np.ndarray, spo2: np.ndarray,
    ) -> float:
        """Compute sleep quality composite score.
        
        sleep_quality = (1 - motion_fraction) * (hr_dipping_ratio) * (1 - spo2_drop_fraction)
        
        Args:
            ax, ay, az: Accelerometer arrays.
            hr: Heart rate array.
            spo2: SpO2 array.
        
        Returns:
            Sleep quality score (0-1).
        """
        # Motion fraction (lower is better for sleep)
        min_acc = min(len(ax), len(ay), len(az))
        if min_acc > 0:
            mag = np.sqrt(ax[:min_acc] ** 2 + ay[:min_acc] ** 2 + az[:min_acc] ** 2)
            motion_threshold = 1.0  # m/s^2
            motion_fraction = np.mean(mag > motion_threshold)
        else:
            motion_fraction = 0.5
        
        # HR dipping ratio (HR should drop during sleep)
        if len(hr) > 0:
            hr_mean = np.mean(hr)
            if hr_mean > self.RESTING_HR_DEFAULT:
                dipping_ratio = self.RESTING_HR_DEFAULT / hr_mean
            else:
                dipping_ratio = 1.0
        else:
            dipping_ratio = 0.8
        
        # SpO2 drop fraction
        if len(spo2) > 0:
            spo2_drop_fraction = np.mean(spo2 < 95)
        else:
            spo2_drop_fraction = 0.0
        
        quality = (1 - motion_fraction) * dipping_ratio * (1 - spo2_drop_fraction)
        return float(np.clip(quality, 0.0, 1.0))
    
    def _compute_fever_score(self, temp: np.ndarray, hr: np.ndarray) -> float:
        """Compute combined fever score.
        
        fever_score = 1.0 if (temp > 38°C AND hr > 100 BPM) else 0.0
        
        Args:
            temp: Temperature array.
            hr: Heart rate array.
        
        Returns:
            Fever score (0 or 1).
        """
        temp_elevated = False
        hr_elevated = False
        
        if len(temp) > 0:
            temp_elevated = np.mean(temp) >= 38.0
        
        if len(hr) > 0:
            hr_elevated = np.mean(hr) > 100.0
        
        return 1.0 if (temp_elevated and hr_elevated) else 0.0
    
    def _compute_fall_risk(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
        hr: np.ndarray,
    ) -> float:
        """Compute fall risk composite score.
        
        fall_risk = movement_unsteadiness * hrv_reduction
        
        Args:
            ax, ay, az: Accelerometer arrays.
            hr: Heart rate array.
        
        Returns:
            Fall risk score (0-1).
        """
        # Movement unsteadiness (variation in movement)
        min_acc = min(len(ax), len(ay), len(az))
        if min_acc > 2:
            mag = np.sqrt(ax[:min_acc] ** 2 + ay[:min_acc] ** 2 + az[:min_acc] ** 2)
            unsteadiness = np.std(mag, ddof=1) / (np.mean(mag) + 0.01)
        else:
            unsteadiness = 0.0
        
        # HRV reduction (low HRV = higher risk)
        if len(hr) > 2:
            hr_std = np.std(hr, ddof=1)
            hrv_factor = max(0.0, 1.0 - hr_std / 10.0)
        else:
            hrv_factor = 0.5
        
        risk = unsteadiness * hrv_factor
        return float(np.clip(risk, 0.0, 1.0))
    
    def _compute_dehydration_proxy(
        self, temp: np.ndarray, hr: np.ndarray, movement_intensity: float
    ) -> float:
        """Compute dehydration proxy.
        
        dehydration_proxy = (temp_elevation * hr_elevation) / movement_intensity
        
        Args:
            temp: Temperature array.
            hr: Heart rate array.
            movement_intensity: Movement intensity value.
        
        Returns:
            Dehydration proxy value.
        """
        if len(temp) == 0 or len(hr) == 0:
            return 0.0
        
        temp_elevation = max(0.0, np.mean(temp) - 36.8)
        hr_elevation = max(0.0, np.mean(hr) - self.RESTING_HR_DEFAULT)
        
        if movement_intensity < 0.01:
            movement_intensity = 0.01
        
        proxy = (temp_elevation * hr_elevation) / movement_intensity
        return float(np.clip(proxy, 0.0, 10.0))
    
    def _compute_cardiovascular_strain(
        self, hr: np.ndarray, spo2: np.ndarray
    ) -> float:
        """Compute cardiovascular strain.
        
        strain = (hr / hr_rest) * (spo2 / 100)
        
        Args:
            hr: Heart rate array.
            spo2: SpO2 array.
        
        Returns:
            Cardiovascular strain value.
        """
        if len(hr) == 0 or len(spo2) == 0:
            return 0.0
        
        hr_mean = np.mean(hr)
        spo2_mean = np.mean(spo2)
        
        if self.RESTING_HR_DEFAULT == 0:
            return 0.0
        
        strain = (hr_mean / self.RESTING_HR_DEFAULT) * (spo2_mean / 100.0)
        return float(strain)
    
    def _compute_recovery_index(
        self, hr: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> float:
        """Compute recovery index (rate of HR decrease after activity cessation).
        
        Args:
            hr: Heart rate array.
            ax, ay, az: Accelerometer arrays.
        
        Returns:
            Recovery index (negative slope indicates recovery).
        """
        if len(hr) < 10:
            return 0.0
        
        min_acc = min(len(ax), len(ay), len(az))
        if min_acc < 10:
            return 0.0
        
        # Compute movement magnitude
        mag = np.sqrt(ax[:min_acc] ** 2 + ay[:min_acc] ** 2 + az[:min_acc] ** 2)
        
        # Find point where movement decreases (activity cessation)
        window_size = max(5, len(mag) // 5)
        rolling_mean = np.convolve(mag, np.ones(window_size) / window_size, mode="valid")
        
        if len(rolling_mean) < 5:
            return 0.0
        
        # Find peak movement
        peak_idx = np.argmax(rolling_mean)
        
        # Look at HR after peak movement
        if peak_idx >= len(hr) - 3:
            return 0.0
        
        hr_after_peak = hr[peak_idx:]
        if len(hr_after_peak) < 3:
            return 0.0
        
        # Compute slope of HR after activity
        t = np.arange(len(hr_after_peak))
        slope, _, _, _, _ = scipy_stats.linregress(t, hr_after_peak)
        
        return float(slope)