import pandas as pd
from .signal_processing import SignalPipeline, generate_quality_report

def main():
    # Load unified dataset
    df = pd.read_parquet('../../data/processed/unified_dataset.parquet')
    
    # Process signals
    pipeline = SignalPipeline()
    processed = pipeline.process_signals(df)
    
    # Save results
    processed.to_parquet('../../data/processed_signals/phase3_clean_signals.parquet')
    generate_quality_report(processed, '../../reports/phase3_quality.csv')

if __name__ == '__main__':
    main()
