import pandas as pd


def generate_quality_report(processed_df: pd.DataFrame, output_path: str):
    """Generate SNR/RMSE metrics comparing to clean baseline"""
    # Get clean baselines
    clean = processed_df[processed_df['device'] == 'clean']
    variants = processed_df[processed_df['device'] != 'clean']
    
    # Merge on timestamps
    merged = variants.merge(
        clean[['bvp_clean']], 
        left_index=True, 
        right_index=True,
        suffixes=('', '_baseline')
    )
    
    # Calculate metrics
    report = merged.groupby(['device', 'skin_tone']).apply(
        lambda g: pd.Series({
            'snr_improvement': 10 * np.log10(g['bvp_clean_baseline'].var() / (g['bvp_clean'] - g['bvp_clean_baseline']).var()),
            'rmse_reduction': np.sqrt(np.mean((g['bvp_clean'] - g['bvp_clean_baseline'])**2)),
            'samples': len(g)
        })
    ).reset_index()
    
    report.to_csv(output_path, index=False)
    return report
