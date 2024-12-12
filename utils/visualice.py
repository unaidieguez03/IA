import polars as pl
import numpy as np

def visualice_lazyframe(lazy_frame, n=5):
    """
    Muestra las primeras `n` filas de un LazyFrame después de materializarlo.

    Parámetros:
        lazy_frame (pl.LazyFrame): El LazyFrame a visualizar.
        n (int): Número de filas a mostrar (por defecto 5).

    Retorna:
        pl.DataFrame: Un DataFrame con las primeras `n` filas.
    """
    if not isinstance(lazy_frame, pl.LazyFrame):
        raise TypeError("El argumento proporcionado no es un LazyFrame.")

    # Materializar el LazyFrame
    df = lazy_frame.collect()
    # Retornar las primeras `n` filas
    return df.head(n)