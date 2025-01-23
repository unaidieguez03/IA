import os
import polars as pl
from tqdm.auto import tqdm

def process_lazy_images(lf: pl.LazyFrame,total_rows:int =0, chunk_size: int = 10000, output_path: str = "output.parquet", name:str="preprocesed_dataset.parquet"):

    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    os.makedirs(output_path, exist_ok=True)
    TEMPORAL_DIR = os.path.join(output_path,f"temp")
    os.makedirs(TEMPORAL_DIR, exist_ok=True)

    for i in  tqdm(range(num_chunks), desc="Parquet creation process"):
        start = i * chunk_size
        temp_file = os.path.join(TEMPORAL_DIR,f"temp_chunk_{i}.parquet")
        (lf
        .slice(start, chunk_size)
        .collect()
        .write_parquet(temp_file))
    combined_lf = pl.scan_parquet([os.path.join(TEMPORAL_DIR,f"temp_chunk_{i}.parquet") for i in range(num_chunks)])

    combined_lf.sink_parquet(
        os.path.join(output_path, name),
        compression="snappy",
        compression_level=22,
    )

    for i in range(num_chunks):
        os.remove(os.path.join(TEMPORAL_DIR,f"temp_chunk_{i}.parquet"))