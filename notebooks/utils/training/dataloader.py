import polars as pl
import torch
from torch.utils.data import Dataset

class LazyFrameDataset(Dataset):
    def __init__(self, lazy_frame: pl.LazyFrame, dataset_length:int = 0):
        self.length = dataset_length
        self.lf = lazy_frame
    
    def __len__(self):
        return self.length
    def _prepare_labels(self, labels):
        condition_to_idx = {
            'Spondylolysthesis': 0,
            'No finding': 1,
            'Surgical implant': 2,
            'Other lesions': 3,
            'Vertebral collapse': 4,
            'Foraminal stenosis': 5,
            'Osteophytes': 6,
            'Disc space narrowing': 7
        }
        
        combined_labels = torch.zeros(len(condition_to_idx))
        
        for condition in labels:
            idx = condition_to_idx[condition]
            combined_labels[idx] = 1.0
        return combined_labels.float()
    def __getitem__(self, idx):
            row = (
                self.lf
                .slice(idx, 1)
                .collect()
                .row(0)
            )
            image = torch.tensor(row[1], dtype=torch.float32)
            image = image.unsqueeze(0)

            labels = self._prepare_labels(row[0])
            return {"image":image, "label":labels}