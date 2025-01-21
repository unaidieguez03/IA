from .early_stoping import EarlyStopping
from .dataloader import LazyFrameDataset
from .checkpoint import ModelCheckpointer
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from optuna import Trial
from pathlib import Path

class Trainer():
    def __init__(self, 
                 autoencoder, 
                 classifier, 
                 patience,
                 checkpointer,
                 training_set: LazyFrameDataset,
                 batch_size: int, 
                 device,
                 n_folds=3):
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.device = device
        self.training_set = training_set
        self.batch_size = batch_size
        self.n_folds = n_folds
        
        self.checkpointer = checkpointer
        self.early_stopping = EarlyStopping(patience=patience, min_delta=10, maximize_metric=True)
        self.optimizer = optim.AdamW(self.autoencoder.parameters(), lr=1e-3)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.loss_history = []
        
    def _signal_tuner(self, trial: Trial, metrics: dict, epoch: int) -> None:
        # Report FNR as the primary metric
        trial.report(metrics['fnr'], epoch)
        if trial.should_prune():
            print(f'Trial pruned at epoch {epoch}')
            # Remove the discard_checkpoint call to preserve the checkpoint
            raise optuna.TrialPruned()

    def train(self, trial: Trial, num_epochs: int) -> None:
        self.autoencoder.to(self.device)
        self.classifier.to(self.device)
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        best_fnr = float('inf')
        best_state = None
        
        try:
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.training_set)):
                print(f'\nFold {fold + 1}/{self.n_folds}')
                
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                
                train_loader = DataLoader(
                    self.training_set,
                    batch_size=self.batch_size,
                    sampler=train_sampler
                )
                val_loader = DataLoader(
                    self.training_set,
                    batch_size=self.batch_size,
                    sampler=val_sampler
                )
                
                for epoch in range(num_epochs):
                    metrics = self._train_fold(train_loader, val_loader)
                    
                    print(f'Epoch [{epoch+1}/{num_epochs}], Fold [{fold+1}/{self.n_folds}]')
                    print(f'Train - Total Loss: {metrics["train"]["total"]:.4f}, FNR: {metrics["train"]["fnr"]:.4f}')
                    print(f'Val - Total Loss: {metrics["val"]["total"]:.4f}, FNR: {metrics["val"]["fnr"]:.4f}')
                    
                    # Save checkpoint before potentially pruning
                    if metrics['train']['fnr'] < best_fnr:
                        best_fnr = metrics['train']['fnr']
                        best_state = {
                            'autoencoder': self.autoencoder.state_dict(),
                            'classifier': self.classifier.state_dict(),
                            'fold': fold,
                            'epoch': epoch,
                            'metrics': metrics
                        }
                        self.checkpointer.save_checkpoint(
                            metric_value=metrics['train']['fnr'],
                            model=self.classifier,
                            message=f"Model saved. New best FNR: {best_fnr} at fold {fold + 1}, epoch {epoch + 1}",
                        )
                    
                    # Signal tuner after saving checkpoint
                    self._signal_tuner(trial, metrics['train'], epoch)
                    
                    if self.early_stopping(metrics['train']['fnr']):
                        print(f'Early stopping triggered at epoch {epoch+1}, fold {fold+1}')
                        break
                    
                    self.loss_history.append(metrics)
                    
        except optuna.TrialPruned:
            # Save final state before pruning
            if best_state is not None:
                print(f"Saving final state before pruning - Best FNR: {best_fnr:.4f}")
                self.checkpointer.save_checkpoint(
                    metric_value=best_fnr,
                    model=self.classifier,
                    message=f"Final state before pruning - Best FNR: {best_fnr}",
                )
            raise
        
        # Load best model state
        if best_state is not None:
            self.autoencoder.load_state_dict(best_state['autoencoder'])
            self.classifier.load_state_dict(best_state['classifier'])
            print(f"\nBest model from fold {best_state['fold'] + 1}, epoch {best_state['epoch'] + 1}")
            print(f"Best FNR: {best_fnr:.4f}")