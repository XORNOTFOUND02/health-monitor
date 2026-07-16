"""
Heart Rate Variability (HRV) feature extraction.

Extracts 18 features from RR intervals derived from PPG data, including:
- Time-domain: Mean RR, SDNN, RMSSD, NN50, pNN50, SDSD, CV, Median RR, Range RR
- Frequency-domain: VLF, LF, HF power, LF/HF ratio, Total power, LF nu, HF nu
- Non-linear: SD1, SD2, SD1/SD2 ratio, Sample entropy

RR intervals are derived from PPG peak detection or approximated from HR.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from .base import BaseFeatureExtractor


class HRVFeatures(BaseFeatureExtractor):
    """Extract HRV features from RR intervals.
    
    This extractor computes time-domain, frequency-domain, and non-linear
    features from inter-beat (RR) intervals within each analysis window.
    
    Features (18 total):
        Time-domain (9):
            hrv_mean_rr, hrv_std_rr (SDNN), hrv_rmssd,
            hrv_nn50, hrv_pnn50, hrv_sdsd,
            hrv_cv, hrv_median_rr, hrv_range_rr
        
        Frequency-domain (7):
            hrv_vlf_power, hrv_lf_power, hrv_hf_power,
            hrv_lf_hf_ratio, hrv_total_power,
            hrv_lf_nu, hrv_hf_nu
        
        Non-linear (4):
            hrv_sd1, hrv_sd2, hrv_sd1_sd2_ratio, hrv_sampen
    
    Note:
        If raw PPG waveform is available, RR intervals are derived via
        peak detection. Otherwise, approximate RR intervals are computed
        from HR BPM: RR_ms = 60000 / HR_BPM.
    
    Frequency bands (based on task force standards):
        - VLF: 0.0033 - 0.04 Hz
        - LF: 0.04 - 0.15 Hz
        - HF: 0.15 - 0.4 Hz
    """
    
    # Frequency band boundaries (Hz)
    VLF_LOW: float = 0.0033
    VLF_HIGH: float = 0.04
    LF_LOW: float = 0.04
    LF_HIGH: float = 0.15
    HF_LOW: float = 0.15
    HF_HIGH: float = 0.4
    
    def __init__(self) -> None:
        """Initialize HRV feature extractor."""
        super().__init__()
        self.feature_names = [
            # Time-domain
            "hrv_mean_rr", "hrv_std_rr", "hrv_rmssd",
            "hrv_nn50", "hrv_pnn50", "hrv_sdsd",
            "hrv_cv", "hrv_median_rr", "hrv_range_rr",
            # Frequency-domain
            "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power",
            "hrv_lf_hf_ratio", "hrv_total_power",
            "hrv_lf_nu", "hrv_hf_nu",
            # Non-linear
            "hrv_sd1", "hrv_sd2", "hrv_sd1_sd2_ratio", "hrv_sampen",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract HRV features from PPG/HR data.
        
        Args:
            window_data: Dictionary containing heart rate data with keys
                'heart_rate' -> 'ppg_raw' (waveform) and/or 'bpm'.
        
        Returns:
            Dictionary of HRV feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Derive RR intervals
        rr_ms = self._derive_rr_intervals(window_data)
        
        if len(rr_ms) < 3:
            # Return default values for all features
            for name in self.feature_names:
                features[name] = 0.0
            return features
        
        # Time-domain features
        features.update(self._extract_time_domain(rr_ms))
        
        # Frequency-domain features
        features.update(self._extract_frequency_domain(rr_ms))
        
        # Non-linear features
        features.update(self._extract_nonlinear(rr_ms))
        
        return features
    
    def _derive_rr_intervals(self, window_data: Dict[str, Any]) -> np.ndarray:
        """Derive RR intervals from PPG or HR data.
        
        Priority:
        1. Peak detection on raw PPG waveform
        2. Approximation from HR BPM values
        
        Args:
            window_data: Window data dictionary.
        
        Returns:
            Array of RR intervals in milliseconds.
        """
        # Try to get raw PPG
        ppg_raw = self._get_sensor_array(window_data, "heart_rate", "ppg_raw")
        
        if len(ppg_raw) > 10:
            # Attempt peak detection on PPG
            rr_from_ppg = self._ppg_to_rr(ppg_raw)
            if len(rr_from_ppg) >= 3:
                return rr_from_ppg
        
        # Fallback: derive from HR BPM
        hr_bpm = self._get_sensor_array(window_data, "heart_rate", "bpm")
        if len(hr_bpm) > 0:
            # Filter out invalid BPM values
            valid_bpm = hr_bpm[(hr_bpm > 30) & (hr_bpm < 220)]
            if len(valid_bpm) > 0:
                rr_ms = 60000.0 / valid_bpm
                return rr_ms
        
        return np.array([], dtype=np.float64)
    
    def _ppg_to_rr(self, ppg: np.ndarray, fs: float = 100.0) -> np.ndarray:
        """Convert PPG waveform to RR intervals using peak detection.
        
        Args:
            ppg: Raw PPG waveform.
            fs: Sampling frequency (Hz). Default 100 Hz.
        
        Returns:
            Array of RR intervals in milliseconds.
        """
        if len(ppg) < 20:
            return np.array([], dtype=np.float64)
        
        def _detect_peaks(x: np.ndarray) -> np.ndarray:
            # Normalize signal
            x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
            
            # Bandpass filter to isolate cardiac signal (0.5-4 Hz)
            nyq = fs / 2.0
            low = 0.5 / nyq
            high = 4.0 / nyq
            if high >= 1.0:
                high = 0.99
            
            try:
                b, a = scipy_signal.butter(2, [low, high], btype="band")
                filtered = scipy_signal.filtfilt(b, a, x_norm)
            except Exception:
                filtered = x_norm
            
            # Find peaks
            min_distance = int(fs * 0.4)  # Minimum 400ms between peaks
            if min_distance < 1:
                min_distance = 1
            
            peaks, _ = scipy_signal.find_peaks(
                filtered,
                height=0.3 * np.std(filtered),
                distance=min_distance,
            )
            return peaks
        
        try:
            peaks = _detect_peaks(ppg)
            if len(peaks) < 2:
                return np.array([], dtype=np.float64)
            
            # Convert peak indices to RR intervals in ms
            rr_samples = np.diff(peaks)
            rr_ms = (rr_samples / fs) * 1000.0
            
            # Filter physiologically plausible RR intervals (300-2000ms)
            valid = (rr_ms > 300) & (rr_ms < 2000)
            return rr_ms[valid]
        except Exception:
            return np.array([], dtype=np.float64)
    
    def _extract_time_domain(self, rr_ms: np.ndarray) -> Dict[str, float]:
        """Extract time-domain HRV features.
        
        Args:
            rr_ms: RR intervals in milliseconds.
        
        Returns:
            Dictionary of time-domain features.
        """
        features: Dict[str, float] = {}
        
        features["hrv_mean_rr"] = self._safe_extract(rr_ms, lambda x: np.mean(x))
        features["hrv_std_rr"] = self._safe_extract(
            rr_ms, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        )
        
        # RMSSD: root mean square of successive differences
        def _rmssd(x: np.ndarray) -> float:
            diffs = np.diff(x)
            return float(np.sqrt(np.mean(diffs ** 2)))
        features["hrv_rmssd"] = self._safe_extract(rr_ms, _rmssd)
        
        # NN50: number of successive differences > 50ms
        def _nn50(x: np.ndarray) -> float:
            diffs = np.abs(np.diff(x))
            return float(np.sum(diffs > 50))
        features["hrv_nn50"] = self._safe_extract(rr_ms, _nn50)
        
        # pNN50: percentage of successive differences > 50ms
        def _pnn50(x: np.ndarray) -> float:
            diffs = np.abs(np.diff(x))
            if len(diffs) == 0:
                return 0.0
            return float(np.sum(diffs > 50) / len(diffs) * 100.0)
        features["hrv_pnn50"] = self._safe_extract(rr_ms, _pnn50)
        
        # SDSD: standard deviation of successive differences
        def _sdsd(x: np.ndarray) -> float:
            diffs = np.diff(x)
            return float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        features["hrv_sdsd"] = self._safe_extract(rr_ms, _sdsd)
        
        # CV: coefficient of variation (SDNN / mean RR)
        def _cv(x: np.ndarray) -> float:
            if len(x) < 2:
                return 0.0
            mean_val = np.mean(x)
            if mean_val == 0:
                return 0.0
            return float(np.std(x, ddof=1) / mean_val)
        features["hrv_cv"] = self._safe_extract(rr_ms, _cv)
        
        features["hrv_median_rr"] = self._safe_extract(rr_ms, lambda x: np.median(x))
        features["hrv_range_rr"] = self._safe_extract(rr_ms, lambda x: np.ptp(x))
        
        return features
    
    def _extract_frequency_domain(self, rr_ms: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain HRV features using Welch's method.
        
        Args:
            rr_ms: RR intervals in milliseconds.
        
        Returns:
            Dictionary of frequency-domain features.
        """
        features: Dict[str, float] = {}
        
        # Initialize defaults
        for key in [
            "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power",
            "hrv_lf_hf_ratio", "hrv_total_power",
            "hrv_lf_nu", "hrv_hf_nu",
        ]:
            features[key] = 0.0
        
        if len(rr_ms) < 10:
            return features
        
        def _freq_features(x: np.ndarray) -> Dict[str, float]:
            result: Dict[str, float] = {}
            
            # Interpolate RR intervals to uniform time series
            # Create time axis from cumulative RR sum
            rr_cumsum = np.cumsum(x)
            t = rr_cumsum - rr_cumsum[0]
            
            # Resample to 4 Hz uniform grid
            fs_resample = 4.0
            t_uniform = np.arange(t[0], t[-1], 1.0 / fs_resample)
            if len(t_uniform) < 16:
                return {k: 0.0 for k in [
                    "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power",
                    "hrv_lf_hf_ratio", "hrv_total_power",
                    "hrv_lf_nu", "hrv_hf_nu",
                ]}
            
            try:
                rr_interp = np.interp(t_uniform, t, x)
            except Exception:
                return {k: 0.0 for k in [
                    "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power",
                    "hrv_lf_hf_ratio", "hrv_total_power",
                    "hrv_lf_nu", "hrv_hf_nu",
                ]}
            
            # Remove trend
            rr_interp = rr_interp - np.mean(rr_interp)
            
            # Welch PSD estimation
            nperseg = min(len(rr_interp), 128)
            if nperseg < 16:
                nperseg = len(rr_interp)
            
            try:
                freqs, psd = scipy_signal.welch(
                    rr_interp,
                    fs=fs_resample,
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                    detrend="linear",
                )
            except Exception:
                return {k: 0.0 for k in [
                    "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power",
                    "hrv_lf_hf_ratio", "hrv_total_power",
                    "hrv_lf_nu", "hrv_hf_nu",
                ]}
            
            # Compute power in each band using trapezoidal integration
            def _band_power(f: np.ndarray, p: np.ndarray, low: float, high: float) -> float:
                mask = (f >= low) & (f <= high)
                if np.sum(mask) < 2:
                    return 0.0
                return float(np.trapz(p[mask], f[mask]))
            
            vlf = _band_power(freqs, psd, self.VLF_LOW, self.VLF_HIGH)
            lf = _band_power(freqs, psd, self.LF_LOW, self.LF_HIGH)
            hf = _band_power(freqs, psd, self.HF_LOW, self.HF_HIGH)
            total = vlf + lf + hf
            
            result["hrv_vlf_power"] = vlf
            result["hrv_lf_power"] = lf
            result["hrv_hf_power"] = hf
            result["hrv_total_power"] = total
            
            # LF/HF ratio
            if hf > 0:
                result["hrv_lf_hf_ratio"] = lf / hf
            else:
                result["hrv_lf_hf_ratio"] = 0.0
            
            # Normalized units
            lf_hf_sum = lf + hf
            if lf_hf_sum > 0:
                result["hrv_lf_nu"] = lf / lf_hf_sum * 100.0
                result["hrv_hf_nu"] = hf / lf_hf_sum * 100.0
            else:
                result["hrv_lf_nu"] = 0.0
                result["hrv_hf_nu"] = 0.0
            
            return result
        
        features = self._safe_extract(rr_ms, _freq_features, default=features)
        return features
    
    def _extract_nonlinear(self, rr_ms: np.ndarray) -> Dict[str, float]:
        """Extract non-linear HRV features (Poincaré and entropy).
        
        Args:
            rr_ms: RR intervals in milliseconds.
        
        Returns:
            Dictionary of non-linear features.
        """
        features: Dict[str, float] = {}
        
        # Poincaré plot features: SD1, SD2
        def _poincare(x: np.ndarray) -> Dict[str, float]:
            if len(x) < 3:
                return {"hrv_sd1": 0.0, "hrv_sd2": 0.0, "hrv_sd1_sd2_ratio": 0.0}
            
            rr_n = x[:-1]
            rr_n1 = x[1:]
            
            # SD1: short-term variability (perpendicular to identity line)
            diff_rr = rr_n1 - rr_n
            sd1 = np.std(diff_rr) / np.sqrt(2)
            
            # SD2: long-term variability (along identity line)
            sum_rr = rr_n1 + rr_n
            sd2 = np.std(sum_rr) / np.sqrt(2)
            
            # SD1/SD2 ratio
            ratio = sd1 / sd2 if sd2 > 0 else 0.0
            
            return {
                "hrv_sd1": float(sd1),
                "hrv_sd2": float(sd2),
                "hrv_sd1_sd2_ratio": float(ratio),
            }
        
        poincare_feats = self._safe_extract(rr_ms, _poincare, default={
            "hrv_sd1": 0.0, "hrv_sd2": 0.0, "hrv_sd1_sd2_ratio": 0.0,
        })
        features.update(poincare_feats)
        
        # Sample entropy
        features["hrv_sampen"] = self._compute_sampen(rr_ms)
        
        return features
    
    def _compute_sampen(
        self, rr_ms: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        """Compute sample entropy of RR intervals.
        
        Args:
            rr_ms: RR intervals in milliseconds.
            m: Embedding dimension.
            r: Tolerance (fraction of std).
        
        Returns:
            Sample entropy value.
        """
        if len(rr_ms) < m + 3:
            return 0.0
        
        def _sampen(x: np.ndarray) -> float:
            n = len(x)
            if n < m + 3:
                return 0.0
            
            # Normalize
            std_x = np.std(x)
            if std_x == 0:
                return 0.0
            x_norm = (x - np.mean(x)) / std_x
            r_val = r * std_x
            
            # Subsample for efficiency
            if n > 150:
                indices = np.linspace(0, n - 1, 150, dtype=int)
                x_norm = x_norm[indices]
                n = len(x_norm)
            
            def _count_matches(data: np.ndarray, dim: int) -> int:
                count = 0
                n_d = len(data)
                for i in range(n_d - dim):
                    template = data[i:i + dim]
                    for j in range(i + 1, n_d - dim):
                        candidate = data[j:j + dim]
                        if np.max(np.abs(template - candidate)) < r_val:
                            count += 1
                return count
            
            A = _count_matches(x_norm, m)
            B = _count_matches(x_norm, m + 1)
            
            if A == 0 or B == 0:
                return 0.0
            
            return float(-np.log(B / A))
        
        return self._safe_extract(rr_ms, _sampen)