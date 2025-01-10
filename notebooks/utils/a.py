@dataclass
class HyperparameterTuner:
    """
    A class for tuning hyperparameters of the port embedding model using Optuna.

    This class manages the hyperparameter optimization process, including model training
    and evaluation across multiple trials. It uses Optuna's optimization framework to
    search the hyperparameter space efficiently.

    Attributes
    ----------
    gc_after_trial : bool
        Whether to run garbage collection after each trial to free memory.

    model_embedding_dimensions : int
        Size of the learned port embeddings.
        Must be positive integer.

    n_jobs : int
        Number of parallel jobs to run during optimization.
        Determines how many trials can run concurrently.

    n_trials : int
        Total number of optimization trials to run.
        Each trial tests a different set of hyperparameters.

    timeout : float | None
        Maximum time in seconds allowed for the entire optimization process.
        None means no time limit.

    sbert_model : str
        The name of the Sentence Transformer model to use for generating embeddings.

    show_progress_bar : bool
        Whether to display a progress bar during optimization.
        Shows trial progress and estimated completion time.

    study_load_if_exists : bool
        Whether to load and continue an existing study with the same name.
        Enables resuming previous optimization runs.

    study_name : str
        Unique identifier for the optimization study.
        Used for saving/loading study data.

    study_pruner : optuna.pruners.BasePruner
        Pruning algorithm for early stopping of unpromising trials.
        Helps reduce computation time on poor hyperparameter sets.

    study_sampler : optuna.samplers.BaseSampler
        Sampling algorithm for suggesting hyperparameter values.
        Determines the strategy for exploring the parameter space.

    study_storage : optuna.storages.BaseStorage
        Storage backend for saving study data.
        Enables persistence and distributed optimization.

    tuning_direction : Literal["minimize", "maximize"]
        Direction for optimization objective.
        - "minimize": Lower values are better (e.g., for loss).
        - "maximize": Higher values are better (e.g., for accuracy).
    """
    gc_after_trial: bool
    model_embedding_dimensions: int
    n_jobs: int
    n_trials: int
    timeout: float | None
    sbert_model: str
    show_progress_bar: bool
    study_load_if_exists: bool
    study_name: str
    study_pruner: optuna.pruners.BasePruner
    study_sampler: optuna.samplers.BaseSampler
    study_storage: optuna.storages.BaseStorage
    tuning_direction: Literal["minimize", "maximize"]

    def _objective(self, trial: Trial) -> float:
        # Define hyperparameter search space
        min_dist = trial.suggest_float('min_dist', 0.0, 0.5)
        n_neighbors = trial.suggest_int('n_neighbors', 15, 100)

        # Create and fit UMAP
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=OUTPUT_EMBEDDINGS_DIM,
            min_dist=min_dist,
            metric="cosine",
            random_state=SEED  # for reproducibility
        )
        
        # Retrieve original embeddings
        original_embeddings = EMBEDDINGS[self.sbert_model]

        # Reduce dimensionality
        reduced_embeddings: NDArray[np.float32] = umap_model.fit_transform(original_embeddings) # type: ignore

        # Define save path
        save_path: Path = CACHE_DIR.joinpath(f"trial_{trial.number}_{self.sbert_model.replace('/', '-')}.npy")

        # Save the reduced embeddings
        np.save(save_path, reduced_embeddings)

        # Save reduced embeddings path
        trial.set_user_attr("embeddings_path", str(save_path))

        trust = trustworthiness(
            original_embeddings,
            reduced_embeddings,
            n_neighbors=n_neighbors,
            metric="cosine"
        )

        trial.set_user_attr("trustworthiness", trust)

        return trust

    def tune(self) -> Study:
        """
        Execute the hyperparameter optimization study.

        Creates or loads an Optuna study and runs the optimization process
        according to the configured parameters.

        Returns
        -------
        Study
            The completed Optuna study containing trial results and statistics.

        Notes
        -----
        The optimization process can be customized through the class parameters:
        - Number of trials and parallel jobs.
        - Time limit.
        - Progress bar visibility.
        - Garbage collection behavior.
        - Study persistence and loading.
        """
        study = create_study(
            storage=self.study_storage,
            sampler=self.study_sampler,
            pruner=self.study_pruner,
            study_name=self.study_name,
            direction=self.tuning_direction,
            load_if_exists=self.study_load_if_exists,
        )

        study.optimize(
            func=self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            catch=(),
            callbacks=None,
            gc_after_trial=self.gc_after_trial,
            show_progress_bar=self.show_progress_bar,
        )

        return study