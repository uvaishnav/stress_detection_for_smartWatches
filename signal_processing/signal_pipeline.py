import pandas as pd
import numpy as np
from tqdm import tqdm
from .motion_removal import AdaptiveFilter
from .wavelet_denoise import WaveletProcessor
from scipy.signal import find_peaks, correlate, resample, peak_prominences
from sklearn.preprocessing import MinMaxScaler

class SignalPipeline:
    def __init__(self):
        self.filter = AdaptiveFilter()
        self.wavelet = WaveletProcessor()
        
    def _process_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Process device/skin-tone variant group"""
        # Separate clean baseline
        clean_df = group[group['device'] == 'clean'][['bvp']].rename(columns={'bvp': 'baseline'})
        
        processed = []
        for device, variants in group.groupby('device'):
            if device == 'clean':
                continue
                
            # Merge with clean baseline using timestamps
            merged = variants.join(clean_df, how='left')
            
            # 1. Motion removal
            acc = variants[['acc_x','acc_y','acc_z']].values
            filtered = self.filter.apply_lms(variants['bvp'].values, acc)
            
            # 2. Wavelet denoising
            skin_tone = variants['skin_tone'].iloc[0]
            cleaned = self.wavelet.denoise(filtered, skin_tone)
            
            # 3. Calculate SNR with aligned baseline
            valid_mask = ~merged['baseline'].isna()
            signal_power = np.nan  # Initialize with default
            noise_power = np.nan    # Initialize with default
            snr_improvement = np.nan
            
            if valid_mask.sum() > 0:
                # Convert pandas Series to numpy arrays before reshaping
                baseline_data = merged['baseline'][valid_mask].values
                cleaned_data = cleaned[valid_mask]
                
                # Before normalization
                cross_corr = correlate(baseline_data, cleaned_data, mode='full')
                best_shift = np.argmax(cross_corr) - len(cleaned_data)
                aligned_cleaned = np.roll(cleaned_data, best_shift)
                
                # Resample to match lengths
                if len(baseline_data) != len(aligned_cleaned):
                    aligned_cleaned = resample(aligned_cleaned, len(baseline_data))
                
                # Normalize both using baseline's parameters
                baseline_offset = np.median(baseline_data)
                cleaned_offset = np.median(aligned_cleaned)
                scaled_baseline = baseline_data - baseline_offset
                scaled_cleaned = aligned_cleaned - cleaned_offset
                
                signal_power = np.mean(scaled_baseline**2)
                noise_power = np.mean((scaled_cleaned - scaled_baseline)**2)
                if noise_power > signal_power:
                    print(f"WARNING: Negative SNR improvement ({snr_improvement:.1f} dB)")
                    print("Possible causes: 1) Misalignment 2) Over-filtering 3) Hardware mismatch")
                snr_improvement = 10 * np.log10(signal_power / (noise_power + 1e-9))  # Prevent div/0
                
            print(f"Aligned baseline samples: {valid_mask.sum()}")
            print(f"Signal power: {signal_power:.4f}, Noise power: {noise_power:.4f}")
            print(f"SNR: {snr_improvement:.1f} dB")
            
            # 4. Store results - use merged DataFrame to keep baseline
            result = merged.copy()
            result['bvp_clean'] = cleaned
            result['bvp_pulse_rate'] = self._compute_pulse_rate(cleaned)
            result['snr_improvement'] = snr_improvement
            processed.append(result.drop(columns=['bvp']))  # Drop original BVP here
            
        return pd.concat(processed)
    
    def _compute_pulse_rate(self, signal: np.ndarray) -> np.ndarray:
        """Robust pulse rate calculation with interpolation"""
        # Find peaks with minimum distance constraint
        dynamic_height = 0.5 * (np.percentile(signal, 75) + np.median(signal))
        peaks, _ = find_peaks(
            signal,
            height=dynamic_height,
            distance=15,  # 30Hz * 0.5s minimum interval
            prominence=0.2
        )
        
        if len(peaks) < 2:
            return np.full(len(signal), np.nan)  # Return NaN array if insufficient peaks
        
        # Calculate instantaneous heart rates
        rates = 60 * 30 / np.diff(peaks)  # 30Hz sampling
        
        # Create time base for interpolation
        rate_times = peaks[:-1] + np.diff(peaks)//2
        time_points = np.arange(len(signal))
        
        # Linear interpolation
        rates = np.interp(time_points, rate_times, rates, left=rates[0], right=rates[-1])
        
        # Add validation
        rates = np.clip(rates, 40, 200)  # Human heart rate limits
        prominences = peak_prominences(signal, peaks)[0]
        valid_peaks = peaks[prominences > 0.3]  # Minimum prominence
        return np.where(np.isnan(rates), 60, rates)  # Fallback to 60 BPM
        
    def process_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point"""
        groups = df.groupby(['subject_id', 'skin_tone'])
        results = [self._process_group(g) for _, g in tqdm(groups)]
        return pd.concat(results)
