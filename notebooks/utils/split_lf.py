import polars as pl
import numpy as np

def split_lazy_df(lazy_df, train_ratio=0.8, seed=42):
    """
    Split a LazyDataFrame into train and test sets.
    
    Args:
        lazy_df: Polars LazyDataFrame to split
        train_ratio: Proportion of data to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_ldf, test_ldf) as LazyDataFrames
    """
    # Generate row numbers and random values
    df_with_rand = lazy_df.with_row_count("row_nr").with_columns(
        pl.lit(np.random.default_rng(seed).random(1)[0]).alias("rand")
    )
    print("q")
    # Split based on random values
    train_ldf = df_with_rand.filter(pl.col("rand") <= train_ratio).drop(["row_nr", "rand"])
    test_ldf = df_with_rand.filter(pl.col("rand") > train_ratio).drop(["row_nr", "rand"])
    
    return train_ldf, test_ldf