from iaModelClasses import classification, encoder
import torch
from pathlib import Path

class classificationModel:
    def __init__(self,save_path: str):
        self.path=Path(save_path)
        self.model=self.load_model(save_path)

    def load_model(path:Path):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Running on the GPU')
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            print('Running on the CPU')
        assert Path.exists(path), "No checkpoint found"
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        modelo1 = classification.Classifier(num_classes=2, encoder=encoder.Encoder())
        modelo1.load_state_dict()
        return checkpoint