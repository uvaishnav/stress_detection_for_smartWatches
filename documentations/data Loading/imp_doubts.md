# 🧑‍💻 **Anti-Aliasing in the Data Loading Module**

#### ❓ **What is Anti-Aliasing?**
Anti-aliasing is a signal processing technique used to prevent **aliasing distortion**, which occurs when high-frequency components in a signal are misrepresented as lower frequencies during downsampling. Without anti-aliasing, critical physiological signal patterns (e.g., BVP waveforms) could be corrupted.

---

#### 🔧 **Implementation in This Module**
The `BaseDataLoader` class implements anti-aliasing via **low-pass filtering** before downsampling. Here’s how it works:

1. **🌀 Butterworth Low-Pass Filter**:  
   - A 5th-order Butterworth filter (`butter_lowpass_filter()`) is applied to the raw signal.  
   - **Cutoff Frequency**: Set to **80% of the Nyquist frequency** of the *target* sampling rate:  
     ```python
     nyquist = target_rate / 2  # e.g., 15 Hz for target_rate=30 Hz
     cutoff = nyquist * 0.8     # 12 Hz in this example
     ```
   - Ensures frequencies above the cutoff are attenuated before resampling.

2. **🔄 Zero-Phase Filtering**:  
   Uses `filtfilt` (forward-backward filtering) to eliminate phase distortion, preserving signal morphology.

3. **⚙️ Resampling Workflow**:  
   ```python
   def resample_data(...):
       if original_rate > target_rate:
           # Apply anti-aliasing filter
           filtered_data = data.apply(lambda x: butter_lowpass_filter(x, ...))
           # Downsample using mean aggregation
           resampled = filtered_data.resample(...).mean()
       return resampled
   ```

---

#### 📊 **Example Scenario**  
For a sensor sampled at **64 Hz** (e.g., BVP in PhysioNet) being downsampled to **30 Hz**:  
1. Nyquist frequency for 30 Hz = 15 Hz.  
2. Cutoff = 15 Hz × 0.8 = **12 Hz**.  
3. All frequencies **above 12 Hz** are filtered out, ensuring no aliasing artifacts after resampling.  

---

#### ⚠️ **Why This Matters**  
> - Preserves **diagnostically relevant signal features** (e.g., heart rate variability in BVP).  
> - Avoids artificial noise that could degrade stress detection model performance.  
> - Follows best practices for biomedical signal processing (IEEE/PhysioNet standards).  

