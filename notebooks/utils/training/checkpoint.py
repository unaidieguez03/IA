from pathlib import Path
import logging
import math
import torch
import torch.nn as nn

__all__: list[str] = [
    "ModelCheckpointer",
]

class ModelCheckpointer:
    """
    A class for managing model checkpoints during training.

    Handles saving, loading, and discarding model checkpoints based on a metric value.
    Maintains only the best checkpoint according to the specified metric direction
    (maximize or minimize).

    Attributes
    ----------
    _maximize_metric : bool
        Whether to maximize or minimize the metric when determining the best checkpoint.
    num_conv_layers
    _best_metric : float
        The best metric value seen so far
        Initialized to negative infinity for maximization, positive infinity for minimization.
    
    _best_checkpoint_path : Path
        Path where the best checkpoint is saved.

    _logger : logging.Logger
        Logger for checkpoint operations.
    """
    def __init__(
        self,
        save_path: str,
        maximize_metric: bool = True
    ):
        """
        Initialize ModelCheckpointer instance.

        Parameters
        ----------
        save_path : Path
            Path where the checkpoint file will be saved.
        maximize_metric : bool, optional
            Whether the metric should be maximized (True) or minimized (False), by default True.
        """
        self._maximize_metric = maximize_metric
        self._best_metric = -math.inf if maximize_metric else math.inf
        self._best_checkpoint_path = save_path
        
        Path.mkdir(
            save_path.parent,
            parents=True,
            exist_ok=True
        )

        self._logger = logging.getLogger(__name__)
    
    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if the current metric value is better than the best seen so far.

        Parameters
        ----------
        current : float
            Current metric value to compare.
        best : float
            Best metric value seen so far.

        Returns
        -------
        bool
            True if current value is better than best, False otherwise.
        """
        return current > best if self._maximize_metric else current < best
    
    def save_checkpoint(
        self,
        metric_value: float,
        model: nn.Module,
        message: str
    ) -> None:

        if not self._is_better(metric_value, self._best_metric):
            return
        
        self._best_metric = metric_value
        checkpoint = {
            "model_state": model.eval().state_dict(),
        }
        torch.save(checkpoint, self._best_checkpoint_path)
    
    def discard_checkpoint(self) -> None:
        """
        Delete the current checkpoint file.

        Used when the training run should be discarded, for example
        when a trial is pruned during hyperparameter optimization.
        """
        self._best_checkpoint_path.unlink(missing_ok=True)
    
    @staticmethod
    def load_best_checkpoint(
        checkpoint_path: Path,
    ):
        """
        Load the best checkpoint from disk.

        Parameters
        ----------
        checkpoint_path : Path
            Path to the checkpoint file.

        Returns
        -------
        Checkpoint
            Dictionary containing:
            - model_state: Model state dictionary.
            - bert_dim: SBERT embedding dimension.
            - history: Training history.

        Raises
        ------
        AssertionError
            If no checkpoint exists at the specified path.
        """
        assert Path.exists(checkpoint_path), "No checkpoint found"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        return checkpoint