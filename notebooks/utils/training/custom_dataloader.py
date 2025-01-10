import polars as pl
import torch
from torch.utils.data import Dataset

class LazyFrameDataset(Dataset):
    def __init__(self, lazy_frame: pl.LazyFrame, target_column:str='image', dataset_length:int = 0):
        self.length = dataset_length
        self.target_column = target_column
        self.lf = lazy_frame
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        row = (
            self.lf
            .select([self.target_column])
            .slice(idx, 1)
            .collect()
            .row(0)
        )
        image = torch.tensor(row[0], dtype=torch.float32)
        image = image.unsqueeze(0)
        return image