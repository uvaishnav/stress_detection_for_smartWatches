
import pandas as pd

class BaseSignalProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the signal processor with the path to the unified dataset.
        
        Parameters:
            data_path (str): Path to the unified Parquet file.
        """
        self.data_path = data_path
        self.dataset = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the unified dataset into memory.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        print("Loading unified dataset...")
        self.dataset = pd.read_parquet(self.data_path)
        return self.dataset