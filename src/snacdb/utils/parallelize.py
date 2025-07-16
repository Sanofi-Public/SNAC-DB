import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import concurrent.futures
import os
from typing import Callable, List, Dict, Tuple, Union
import traceback

class ParallelProcessorForCPUBoundTasks:
    def __init__(self, process_row: Callable, max_workers: int = 4, verbose: bool = True, chunk_size: int = None):
        """
        A parallel processor optimized for CPU-bound tasks using process-based parallelism and data chunking.

        Args:
            process_row (Callable): Function to process a single row. 
                                    Must accept a row (dict or tuple) and additional args/kwargs.
            max_workers (int): Maximum number of worker processes for parallel processing.
            verbose (bool): Whether to show progress with tqdm.
            chunk_size (int, optional): Number of rows per chunk. If None, it's auto-calculated based on the data size.
        """
        self.process_row = process_row
        self.max_workers = max_workers
        self.verbose = verbose
        self.chunk_size = chunk_size

    def _process_chunk(self, chunk: List[Union[Dict, Tuple]], *args, **kwargs) -> List:
        """
        Process a chunk of data rows.

        Args:
            chunk (List[Union[Dict, Tuple]]): A chunk of data rows.
        
        Returns:
            List: Processed rows (excluding None results).
        """
        processed_rows = []
        for row in chunk:
            try:
                result = self.process_row(row, *args, **kwargs)
                if result is not None:
                    processed_rows.append(result)
            except Exception as e:
                print(f"Error in row: {row}")
                traceback.print_exc()  # Show full traceback
        return processed_rows

    def _split_data(self, data: List[Union[Dict, Tuple]], num_splits: int) -> List[List[Union[Dict, Tuple]]]:
        """
        Split data into approximately equal-sized chunks for parallel processing.

        Args:
            data (List[Union[Dict, Tuple]]): The dataset to split.
            num_splits (int): Number of splits (based on max_workers).

        Returns:
            List[List]: A list of data chunks.
        """
        chunks = [[] for _ in range(num_splits)]  # Initialize empty lists for each chunk

        # Distribute items to chunks in a round-robin fashion
        for idx, item in enumerate(data):
            chunks[idx % num_splits].append(item)

        return chunks

    def process(self, data: Union[List[Dict], List[Tuple]], *args, **kwargs) -> List:
        """
        Process rows of data in parallel using process-based parallelism with chunking.

        Args:
            data (Union[List[Dict], List[Tuple]]): List of data rows.
            *args: Positional arguments to pass to `process_row`.
            **kwargs: Keyword arguments to pass to `process_row`.

        Returns:
            List: All processed rows (excluding None results).
        """
        # Determine chunk size dynamically if not provided
        num_chunks = self.max_workers
        data_chunks = self._split_data(data, num_chunks)

        processed_rows = []

        # Use ProcessPoolExecutor for CPU-bound parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk, *args, **kwargs) for chunk in data_chunks]

            # Collect results with tqdm progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), 
                               disable=not self.verbose, 
                               desc="Processing Chunks"):
                try:
                    processed_rows.extend(future.result())
                except Exception as e:
                    print(f"Error in parallel processing: {e}")

        return processed_rows