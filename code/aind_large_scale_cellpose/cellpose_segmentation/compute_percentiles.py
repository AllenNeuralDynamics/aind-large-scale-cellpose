"""
Computes global percentiles in the whole dataset.
"""

import time
from typing import Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
from aind_large_scale_prediction._shared.types import ArrayLike
from aind_large_scale_prediction.generator.utils import concatenate_lazy_data
from aind_large_scale_prediction.generator.zarr_slice_generator import BlockedZarrArrayIterator
from aind_large_scale_prediction.io import extract_data
from dask import config as da_cfg
from distributed import Client, LocalCluster


def set_dask_config(dask_folder: str):
    """
    Sets dask configuration

    Parameters
    ----------
    dask_folder: str
        Path to the temporary directory and local directory
        of workers in dask.
    """
    # Setting dask configuration
    da_cfg.set(
        {
            "temporary-directory": dask_folder,
            "local_directory": dask_folder,
            # "tcp-timeout": "300s",
            "array.chunk-size": "128MiB",
            "distributed.worker.memory.target": 0.90,  # 0.85,
            "distributed.worker.memory.spill": 0.92,  # False,#
            "distributed.worker.memory.pause": 0.95,  # False,#
            "distributed.worker.memory.terminate": 0.98,
        }
    )


def get_channel_percentiles(
    array: ArrayLike,
    block_shape: Tuple[int],
    percentile_range: Tuple[float, float],
    min_cell_volume: Optional[float] = 0,
):
    """
    Partitions the last 3 dimensions of a Dask array
    into non-overlapping blocks and computes the percentile.

    Parameters
    ----------
    array: ArrayLike
        Array to process.

    block_shape: Tuple[int]
        Chunk of data to process

    percentile_range: Tuple[float, float]
        Percentile range to compute.

    min_cell_volume: Optional[float]
        Minimum value in the array data.
        Default = 0.

    Returns
    -------
    dict
        Dictionary with the computed percentiles.
    """
    # Iterate through the input array in steps equal to the block shape dimensions
    slices_to_process = list(BlockedZarrArrayIterator.gen_slices(array.shape, block_shape))

    percentiles = {}

    for sl in slices_to_process:

        block = array[sl]
        block = block[block > min_cell_volume]
        percentiled_block = da.percentile(block, percentile_range, method="linear")
        min_max_values = percentiled_block.compute()

        percentiles[str(sl)] = min_max_values

    return percentiles


def compute_chunked_percentiles(
    lazy_data: ArrayLike,
    target_size_mb: int,
    percentile_range: Tuple[float, float],
    min_cell_volume: Optional[int] = 0,
    n_workers: Optional[int] = 0,
    threads_per_worker: Optional[int] = 1,
) -> Dict:
    """
    Computes the percentile in a dataset using chunks.

    Parameters
    ----------
    lazy_data: ArrayLike
        Loaded lazy array. This could be a multichannel
        lazy array in which case, percentiles will be computed
        per channel.

    target_size_mb: int
        Size to fit the current data in memory.

    percentile_range: Tuple[float, float]
        Percentile range to compute.

    min_cell_volume: Optional[int]
        Minimum voxel value. Default: 0

    n_workers: Optional[int]
        Number of workers. Recommended to number of CPUs.
        Default: 0

    threads_per_worker: Optional[int]
        Threads per worker. Default: 1

    Returns
    -------
    Dict
        Dictionary with the computed percentiles per channel.
        Each channel contains keys to the loaded chunks and
        computed percentiles.
    """

    # Squeezing the data
    lazy_data = extract_data(lazy_data)

    if lazy_data.ndim == 3:
        lazy_data = da.expand_dims(lazy_data, axis=0)

    # Instantiating local cluster for parallel writing
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)

    percentiles = {}
    for ch_axis in range(lazy_data.shape[-4]):

        print(f"Processing channel: {ch_axis} - Cluster: {cluster}")
        chn_lazy_data = lazy_data[ch_axis, ...]
        # Block shape to process
        block_shape = list(
            BlockedZarrArrayIterator.get_block_shape(
                arr=chn_lazy_data, target_size_mb=target_size_mb
            )
        )

        # Computes global percentiles
        chn_percentiles = get_channel_percentiles(
            array=chn_lazy_data,
            block_shape=block_shape,
            percentile_range=percentile_range,
            min_cell_volume=min_cell_volume,
        )
        percentiles[ch_axis] = chn_percentiles

    client.shutdown()

    return percentiles


def combine_percentiles(percentiles: Dict, method: Optional[str] = "min_max") -> List:
    """
    Combines the percentiles per channel.

    Parameters
    ----------
    percentiles: Dict
        Dictionary with the computed percentiles per channel.
        Each channel contains keys to the loaded chunks and
        computed percentiles.

    method: Optional[str]
        Method to get the final percentile value.
        Accepted values are: ["min_max", "median"]

    Returns
    -------
    Dict
        Dictionary with the global percentiles per channel.
    """

    method = method.lower()
    if method not in ["min_max", "median"]:
        raise NotImplementedError(f"Method {method} not currently implemented.")

    combined_percentiles = []
    for chn_idx, chunked_percentiles in percentiles.items():
        channel_percentiles = np.array(list(chunked_percentiles.values())).T

        channel_percentiles_cmb = None
        if method == "min_max":
            channel_percentiles_cmb = np.array(
                [
                    np.min(channel_percentiles[0]),
                    np.max(channel_percentiles[1]),
                ]
            )

        elif method == "median":
            channel_percentiles_cmb = np.median(channel_percentiles, axis=1)

        combined_percentiles.append(list(channel_percentiles_cmb))

        print(f"Channel {chn_idx}: {channel_percentiles} - cmb: {channel_percentiles_cmb}")

    return combined_percentiles


def compute_percentiles(
    lazy_data: ArrayLike,
    target_size_mb: int,
    percentile_range: Tuple[float, float],
    dask_folder: str,
    min_cell_volume: Optional[int] = 0,
    n_workers: Optional[int] = 0,
    threads_per_worker: Optional[int] = 1,
    combine_method: Optional[str] = "median",
) -> Tuple[Dict, Dict]:
    """
    Computes the global percentile in a dataset.

    Parameters
    ----------
    lazy_data: ArrayLike
        Loaded lazy array. This could be a multichannel
        lazy array in which case, percentiles will be computed
        per channel.

    target_size_mb: int
        Size to fit the current data in memory.

    percentile_range: Tuple[float, float]
        Percentile range to compute.

    dask_folder: str
        Path to the dask default folder.

    min_cell_volume: Optional[int]
        Minimum voxel value. Default: 0

    n_workers: Optional[int]
        Number of workers. Recommended to number of CPUs.
        Default: 0

    threads_per_worker: Optional[int]
        Threads per worker. Default: 1

    combine_method: Optional[str]
        Combination method for the percentiles.
        Default: 'median'

    Returns
    -------
    Dict
        Dictionary with the computed percentiles per channel.
        Each channel contains keys to the loaded chunks and
        computed percentiles.
    """
    set_dask_config(dask_folder=dask_folder)

    percentiles = compute_chunked_percentiles(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        percentile_range=percentile_range,
        min_cell_volume=min_cell_volume,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )

    combined_percentiles = combine_percentiles(percentiles=percentiles, method=combine_method)

    return combined_percentiles, percentiles


def main():
    """Main function to compute percentiles"""

    dataset_path = "/path/to/channel_405.zarr"
    nuclear_channel = "/path/to/channel/channel_3.zarr"
    multiscale = "2"
    target_size_mb = 4096
    n_workers = 10

    lazy_data = concatenate_lazy_data(
        dataset_paths=[dataset_path, nuclear_channel],
        multiscales=[multiscale, multiscale],
        concat_axis=-4,
    )

    start_time = time.time()
    combined_percentiles, chunked_percentiles = compute_percentiles(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        percentile_range=(10, 99),
        min_cell_volume=95,
        n_workers=n_workers,
        threads_per_worker=1,
        combine_method="median",
    )

    end_time = time.time()

    print(f"Time to compute percentiles: {end_time - start_time}")
    print(f"Percentiles: {chunked_percentiles}")

    print(f"Combined percentiles: {combined_percentiles}")


if __name__ == "__main__":
    main()
