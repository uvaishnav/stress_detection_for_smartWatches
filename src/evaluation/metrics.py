# Old thresholds (invalid)
# ACCEPTABLE_SNR = 20  # dB  

# New thresholds based on calibrated noise
SKIN_TONE_SNR_THRESHOLDS = {
    'light': 18.0,  # 24.4dB measured → safe margin
    'medium': 15.0, # 19.2dB measured
    'dark': 12.0    # 14.3dB measured
}
