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
        
    def calculate_fnr(self, predictions, labels):
        predictions = (torch.sigmoid(predictions) > 0.5).float()
        false_negatives = ((predictions == 0) & (labels == 1)).sum().item()
        actual_positives = (labels == 1).sum().item()
        fnr = false_negatives / actual_positives if actual_positives > 0 else 0
        return fnr

    def _signal_tuner(self, trial: Trial, metrics: dict, epoch: int) -> None:
        # Report FNR as the primary metric
        trial.report(metrics['fnr'], epoch)
        if trial.should_prune():
            print(f'prune')
            raise optuna.TrialPruned()

    def _train_fold(self, train_loader, val_loader) -> dict:
        self.autoencoder.train()
        self.classifier.train()
        
        train_losses = {'total': 0.0, 'reconstruction': 0.0, 'classification': 0.0, 'fnr': 0.0}
        val_losses = {'total': 0.0, 'reconstruction': 0.0, 'classification': 0.0, 'fnr': 0.0}
        
        # Training phase
        for data in tqdm(train_loader, desc="Training"):
            batch = data["image"].to(self.device)
            label = data["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            reconstructed = self.autoencoder(batch)
            reconstruction_loss = self.criterion1(reconstructed["recontruction"], batch)
            # print(label, reconstruction_loss.item(), batch)
            prediction = self.classifier(batch)
            classification_loss = self.criterion2(prediction, label)
            fnr = self.calculate_fnr(prediction, label)
            
            total_loss = reconstruction_loss + classification_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            train_losses['total'] += total_loss.item()
            train_losses['reconstruction'] += reconstruction_loss.item()
            train_losses['classification'] += classification_loss.item()
            train_losses['fnr'] += fnr
            # print(train_losses)
        
        # Validation phase
        self.autoencoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            for data in val_loader:
                batch = data["image"].to(self.device)
                label = data["label"].to(self.device)
                
                reconstructed = self.autoencoder(batch)
                reconstruction_loss = self.criterion1(reconstructed["recontruction"], batch)
                
                prediction = self.classifier(batch)
                classification_loss = self.criterion2(prediction, label)
                fnr = self.calculate_fnr(prediction, label)

                total_loss = reconstruction_loss + classification_loss
                
                val_losses['total'] += total_loss.item()
                val_losses['reconstruction'] += reconstruction_loss.item()
                val_losses['classification'] += classification_loss.item()
                val_losses['fnr'] += fnr
        
        # Average the losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
            val_losses[key] /= len(val_loader)
            
        return {'train': train_losses, 'val': val_losses}

    def train(self, trial: Trial, num_epochs: int) -> None:
        self.autoencoder.to(self.device)
        self.classifier.to(self.device)
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        best_fnr = float('inf')
        best_state = None
        
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
                
                self._signal_tuner(trial, metrics['train'], epoch)
                
                if metrics['train']['fnr'] < best_fnr:
                    best_fnr = metrics['train']['fnr']
                    best_state = {
                        'autoencoder': self.autoencoder.state_dict(),
                        'classifier': self.classifier.state_dict(),
                        'fold': fold,
                        'epoch': epoch
                    }
                
                self.checkpointer.save_checkpoint(
                    metric_value=best_fnr,
                    model=best_state["classifier"],
                    message=f"Model saved. New best FNR: {best_fnr} at fold {fold + 1}, epoch {epoch + 1}",
                )
                
                if self.early_stopping(metrics['train']['fnr']):
                    print(f'Early stopping triggered at epoch {epoch+1}, fold {fold+1}')
                    break
                print(metrics)
                self.loss_history.append(metrics)
        
        # Load best model state
        if best_state is not None:
            self.autoencoder.load_state_dict(best_state['autoencoder'])
            self.classifier.load_state_dict(best_state['classifier'])
            print(f"\nBest model from fold {best_state['fold'] + 1}, epoch {best_state['epoch'] + 1}")
            print(f"Best FNR: {best_fnr:.4f}")

    def free(self) -> None:
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
