import pandas as pd
from pathlib import Path
from typing import Dict, List
from .sensor_alignment import SensorAligner
from .physionet_preparer import PhysioNetPreparer
from .wesad_preparer import WESADPreparer
import logging
import numpy as np

class UnifiedDataPipeline:
    def __init__(self, 
               raw_data_dir: Path = Path('raw_data'),
               output_dir: Path = Path('data/processed')):
        self.output_dir = output_dir
        self.raw_data_dir = raw_data_dir
        self.aligner = SensorAligner()
        self.logger = logging.getLogger('DataPipeline')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'pipeline.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.physionet_exams = ['Midterm 1', 'Midterm 2', 'Final']

    def process_subject(self, subject_id: int, datasets: List[str]) -> Dict:
        """Process data from multiple datasets for a single subject"""
        results = {}
        self.logger.debug(f"Starting processing for subject {subject_id}")
        
        for dataset in datasets:
            try:
                self.logger.debug(f"Processing {dataset} for subject {subject_id}")
                if dataset == 'physionet':
                    # Initialize with correct paths
                    preparer = PhysioNetPreparer(
                        data_path=self.raw_data_dir / 'a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0/DATA',
                        output_dir=self.output_dir
                    )
                    for exam in self.physionet_exams:
                        result = self._process_physionet_exam(preparer, subject_id, exam)
                        if result['status'] == 'success':
                            self.logger.debug(f"Successfully processed {dataset}, files: {result['files']}")
                            aligned = self._align_datasets(result['files'], dataset)
                            results[f"{dataset}_{exam}"] = aligned
                elif dataset == 'wesad':
                    if 2 <= subject_id <= 17:
                        # Initialize with correct paths
                        preparer = WESADPreparer(
                            data_path=self.raw_data_dir / 'WESAD',
                            output_dir=self.output_dir
                        )
                        result = preparer.process_subject(subject_id)
                        if result['status'] == 'success':
                            self.logger.debug(f"Successfully processed {dataset}, files: {result['files']}")
                            aligned = self._align_datasets(result['files'], dataset)
                            results[dataset] = aligned
                else:
                    raise ValueError(f"Unsupported dataset: {dataset}")
                    
            except Exception as e:
                self.logger.debug(f"Exception during {dataset} processing: {str(e)}", exc_info=True)
                results[dataset] = {
                    'status': 'error',
                    'error': str(e),
                    'files': []
                }
        
        self.logger.debug(f"Final results count for subject {subject_id}: {len(results)}")
        return results if results else {'status': 'error', 'error': 'No datasets processed'}

    def _process_physionet_exam(self, preparer: PhysioNetPreparer, 
                              subject_id: int, exam_type: str) -> Dict:
        """Handle PhysioNet exam with tags.csv validation"""
        try:
            # Check if tags file exists and is non-empty
            if preparer.loader.is_tags_empty(subject_id, exam_type):
                self.logger.info(f"Skipping {exam_type} - empty tags.csv")
                return {'status': 'skipped', 'error': 'Empty tags file'}
            
            return preparer.process_subject(subject_id, exam_type)
        except Exception as e:
            self.logger.error(f"PhysioNet {exam_type} error: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def _align_datasets(self, file_paths: List[str], dataset: str) -> Dict[str, pd.DataFrame]:
        """Save aligned variants and return their paths"""
        variants = {}
        output_dir = self.output_dir / 'aligned_variants'
        output_dir.mkdir(exist_ok=True)
        
        self.logger.debug(f"Aligning {len(file_paths)} files for {dataset}")
        
        for path in file_paths:
            df = pd.read_parquet(path)
            self.logger.debug(f"Aligning {path} (shape: {df.shape})")
            
            key = f"{df['device'].iloc[0]}_{df['skin_tone'].iloc[0]}"
            aligned = self.aligner.temporal_align(
                self.aligner.align_acc(df, dataset),
                reference=None
            )
            
            aligned_path = output_dir / f"aligned_{Path(path).name}"
            self.logger.debug(f"Saving aligned data to {aligned_path}")
            aligned.to_parquet(aligned_path)
            
            if aligned.empty:
                self.logger.warning(f"Empty aligned dataframe for {path}")
            
            variants[key] = str(aligned_path)
        
        return variants

    def _merge_files(self, dataset_results: Dict) -> pd.DataFrame:
        """Merge from saved aligned files"""
        all_dfs = []
        required_columns = [
            'bvp', 'label', 'subject_id', 'dataset',
            'device', 'skin_tone', 'noise_level',
            'acc_x', 'acc_y', 'acc_z'
        ]
        
        self.logger.debug(f"Starting merge of {len(dataset_results)} dataset results")
        
        # Collect all aligned file paths
        for source, variants in dataset_results.items():
            if isinstance(variants, dict):
                self.logger.debug(f"Processing {source} with {len(variants)} variants")
                for path in variants.values():
                    df = pd.read_parquet(path)
                    
                    # Add dataset source and filter columns
                    df['dataset'] = source.split('_')[0]  # Extract 'physionet' or 'wesad'
                    df = df[df.columns.intersection(required_columns)]
                    
                    # Convert timestamps to UTC
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    else:
                        df.index = df.index.tz_convert('UTC')
                    
                    all_dfs.append(df)

        self.logger.debug(f"Merging {len(all_dfs)} dataframes")
        merged = pd.concat(all_dfs, axis=0).sort_index()
        
        # Ensure final columns match requirements
        final_cols = [col for col in required_columns if col in merged.columns]
        return merged[final_cols]

    def run_batch(self, subject_ids: List[int], datasets: List[str]):
        """Updated to use new merging"""
        dataset_results = {}
        
        self.logger.info(f"Starting batch processing of {len(subject_ids)} subjects")
        
        for sid in subject_ids:
            valid_datasets = [d for d in datasets if self._is_valid_subject(sid, d)]
            self.logger.debug(f"Processing subject {sid} with datasets: {valid_datasets}")
            result = self.process_subject(sid, valid_datasets)
            dataset_results.update(result)
            self.logger.debug(f"Current result count: {len(dataset_results)}")
        
        unified = self._merge_files(dataset_results)
        unified_path = self.output_dir / 'unified_dataset.parquet'
        self.logger.info(f"Saving unified dataset to {unified_path} with {len(unified)} rows")
        unified.to_parquet(unified_path)

    def _is_valid_subject(self, subject_id: int, dataset: str) -> bool:
        """Check validity for a specific dataset"""
        if dataset == 'wesad':
            return 2 <= subject_id <= 17
        if dataset == 'physionet':
            return 1 <= subject_id <= 10
        return False 