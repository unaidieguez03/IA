class EarlyStopping:
    def __init__(self, patience: int | None = None,
        min_delta: float = 0.0,
        maximize_metric: bool = False):

        self.patience = patience
        self.min_delta = min_delta
        self.maximize_metric = maximize_metric
        self.counter: int = 0
        self.best_metric: float | None = None
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.maximize_metric:
            return current > best + self.min_delta
        return current < best - self.min_delta
    def __call__(self, train_loss):
        if self.best_metric is None:
            self.best_metric = train_loss
        elif not self._is_improvement(train_loss, self.best_metric):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_metric = train_loss
            self.counter = 0
        return False
