import os
import polars as pl

def process_lazy_images(lf: pl.LazyFrame,total_rows:int =0, chunk_size: int = 10000, output_path: str = "output.parquet"):
    """
    Process a LazyFrame containing images in chunks and save to parquet.
    
    Args:
        lf (pl.LazyFrame): Input LazyFrame containing image data
        chunk_size (int): Number of rows to process in each chunk
        output_path (str): Path to save the final parquet file
    """
    # Get total number of rows
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    os.makedirs(output_path, exist_ok=True)
    TEMPORAL_DIR = os.path.join(output_path,f"temp")
    os.makedirs(TEMPORAL_DIR, exist_ok=True)

    for i in range(num_chunks):
        # Calculate chunk bounds
        start = i * chunk_size
        # Process chunk and save to temporary parquet
        temp_file = os.path.join(TEMPORAL_DIR,f"temp_chunk_{i}.parquet")
        (lf
        .slice(start, chunk_size)
        .collect()
        .write_parquet(temp_file))

        print(f"Processed and saved chunk {i + 1}/{num_chunks}")
    
    # Combine all chunks efficiently using Polars
    combined_lf = pl.scan_parquet([os.path.join(TEMPORAL_DIR,f"temp_chunk_{i}.parquet") for i in range(num_chunks)])
    
    # # Write final output
    # combined_lf = combined_lf.explode("lesion_type")

    combined_lf.sink_parquet(
        os.path.join(output_path,"p.parquet"),
        compression="snappy",
        compression_level=22,
    )
    #.write_parquet(output_path)
    
    # Clean up temp files
    for i in range(num_chunks):
        os.remove(os.path.join(TEMPORAL_DIR,f"temp_chunk_{i}.parquet"))