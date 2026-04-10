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


def get_histogram_range(
    dtype: np.dtype,
    n_bins_cap: int = 65536,
) -> Tuple[float, float, int]:
    """
    Returns (value_min, value_max, n_bins) for np.histogram given a numpy dtype.

    For integer dtypes the full representable range is used, capped at
    n_bins_cap bins. For float dtypes a [0.0, 1.0] range is assumed.

    Parameters
    ----------
    dtype : np.dtype
        The data type of the array.
    n_bins_cap : int
        Maximum number of histogram bins. Default: 65536.

    Returns
    -------
    Tuple[float, float, int]
        (value_min, value_max, n_bins)
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        full_range = int(info.max) - int(info.min) + 1
        n_bins = min(full_range, n_bins_cap)
        return float(info.min), float(info.max), n_bins
    else:
        # float32 / float64: assume data lives in [0, 1]
        return 0.0, 1.0, n_bins_cap


def compute_channel_histogram(
    array: ArrayLike,
    block_shape: Tuple[int, ...],
    n_bins: int,
    value_range: Tuple[float, float],
    min_cell_volume: Optional[float] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulates a single global histogram over a 3-D channel array by
    summing per-block histograms.

    Histograms are composable: summing per-block bin counts is equivalent
    to histogramming the entire array at once, giving exact percentile
    estimates (within one bin width).

    Parameters
    ----------
    array : ArrayLike
        3-D channel array (z, y, x).
    block_shape : Tuple[int, ...]
        Block shape used to partition the array.
    n_bins : int
        Number of histogram bins.
    value_range : Tuple[float, float]
        (min_val, max_val) passed to np.histogram as range.
    min_cell_volume : Optional[float]
        Voxels with value <= min_cell_volume are excluded (background
        filter). Default: 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (counts, bin_edges) where counts has shape (n_bins,) dtype
        np.int64 and bin_edges has shape (n_bins + 1,).
    """
    slices_to_process = list(BlockedZarrArrayIterator.gen_slices(array.shape, block_shape))

    global_counts = np.zeros(n_bins, dtype=np.int64)

    # Compute bin_edges once so all blocks use identical boundaries,
    # guaranteeing element-wise summability.
    _, bin_edges = np.histogram(np.array([], dtype=np.float64), bins=n_bins, range=value_range)

    for sl in slices_to_process:
        try:
            block = np.asarray(array[sl]).ravel()
            block = block[block > min_cell_volume]

            if block.size == 0:
                continue

            counts, _ = np.histogram(block, bins=bin_edges)
            global_counts += counts.astype(np.int64)

        except Exception as e:
            print(f"Problem {e} computing histogram in area {sl}")
            continue

    return global_counts, bin_edges


def percentiles_from_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    percentile_range: Tuple[float, float],
) -> List[float]:
    """
    Compute exact percentile values from a histogram via empirical CDF.

    Parameters
    ----------
    counts : np.ndarray
        Histogram bin counts, shape (n_bins,), dtype np.int64.
    bin_edges : np.ndarray
        Bin edges from np.histogram, shape (n_bins + 1,).
    percentile_range : Tuple[float, float]
        (p_low, p_high) where each value is in [0, 100].

    Returns
    -------
    List[float]
        [value_at_p_low, value_at_p_high]

    Raises
    ------
    ValueError
        If counts sums to zero (channel is empty or fully filtered).
    """
    total = counts.sum()
    if total == 0:
        raise ValueError(
            "Histogram is empty — all voxels were filtered by min_cell_volume "
            "or the channel contains no data."
        )

    cdf = np.cumsum(counts) / float(total)

    result = []
    for p in percentile_range:
        idx = int(np.searchsorted(cdf, p / 100.0, side="left"))
        idx = min(idx, len(bin_edges) - 2)
        result.append(float(bin_edges[idx]))

    return result


def compute_chunked_percentiles(
    lazy_data: ArrayLike,
    target_size_mb: int,
    percentile_range: Tuple[float, float],
    min_cell_volume: Optional[int] = 0,
) -> Dict:
    """
    Computes exact global percentiles per channel using histogram accumulation.

    Per-block histograms are summed element-wise into a single global
    histogram, then exact percentile values are derived from the empirical
    CDF. This is correct for any data distribution, including volumes with
    non-uniform spatial intensity (e.g. signal-dense vs. background regions).

    Parameters
    ----------
    lazy_data: ArrayLike
        Loaded lazy array. This could be a multichannel lazy array in which
        case percentiles are computed per channel.
    target_size_mb: int
        Target block size in MB for iterating the array.
    percentile_range: Tuple[float, float]
        (p_low, p_high) percentile pair to compute.
    min_cell_volume: Optional[int]
        Background threshold; voxels <= this value are excluded. Default: 0.

    Returns
    -------
    Dict
        {
          channel_idx: {
            "histogram": (counts: np.ndarray, bin_edges: np.ndarray),
            "percentiles": [p_low_value, p_high_value],
          }
        }
    """
    lazy_data = extract_data(lazy_data)

    if lazy_data.ndim == 3:
        lazy_data = da.expand_dims(lazy_data, axis=0)

    result = {}
    for ch_axis in range(lazy_data.shape[-4]):
        print(f"Processing channel: {ch_axis}")
        chn_lazy_data = lazy_data[ch_axis, ...]

        block_shape = list(
            BlockedZarrArrayIterator.get_block_shape(
                arr=chn_lazy_data, target_size_mb=target_size_mb
            )
        )

        value_min, value_max, n_bins = get_histogram_range(chn_lazy_data.dtype)

        counts, bin_edges = compute_channel_histogram(
            array=chn_lazy_data,
            block_shape=block_shape,
            n_bins=n_bins,
            value_range=(value_min, value_max),
            min_cell_volume=min_cell_volume,
        )

        chn_percentiles = percentiles_from_histogram(
            counts=counts,
            bin_edges=bin_edges,
            percentile_range=percentile_range,
        )

        print(f"Channel {ch_axis} percentiles: {chn_percentiles}")
        result[ch_axis] = {
            "histogram": (counts, bin_edges),
            "percentiles": chn_percentiles,
        }

    return result


def compute_percentiles(
    lazy_data: ArrayLike,
    target_size_mb: int,
    percentile_range: Tuple[float, float],
    dask_folder: str,
    min_cell_volume: Optional[int] = 0,
    n_workers: Optional[int] = 0,
    threads_per_worker: Optional[int] = 1,
) -> Tuple[List, Dict]:
    """
    Computes exact global percentiles per channel using histogram accumulation.

    Parameters
    ----------
    lazy_data: ArrayLike
        Loaded lazy array. This could be a multichannel lazy array in which
        case percentiles are computed per channel.
    target_size_mb: int
        Target block size in MB for iterating the array.
    percentile_range: Tuple[float, float]
        (p_low, p_high) percentile pair to compute.
    dask_folder: str
        Path for dask temporary storage (used only for dask config).
    min_cell_volume: Optional[int]
        Background threshold; voxels <= this value are excluded. Default: 0.
    n_workers: Optional[int]
        Kept for API compatibility. Currently unused. Default: 0.
    threads_per_worker: Optional[int]
        Kept for API compatibility. Currently unused. Default: 1.
    combine_method: Optional[str]
        Deprecated. Previously selected the aggregation strategy for
        approximate per-block percentile estimates. The histogram approach
        computes exact global percentiles directly; this parameter is
        accepted but ignored. Default: 'median'.

    Returns
    -------
    Tuple[List, Dict]
        combined_percentiles : List[List[float]]
            [[p_low_ch0, p_high_ch0], [p_low_ch1, p_high_ch1], ...]
            Ready for use in percentile_normalization().
        chunked_percentiles : Dict
            Full per-channel histogram dict from
            compute_chunked_percentiles, keyed by channel index.
    """

    set_dask_config(dask_folder=dask_folder)

    chunked_percentiles = compute_chunked_percentiles(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        percentile_range=percentile_range,
        min_cell_volume=min_cell_volume,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )

    combined_percentiles = [
        chunked_percentiles[ch_idx]["percentiles"] for ch_idx in sorted(chunked_percentiles.keys())
    ]

    return combined_percentiles, chunked_percentiles


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
    )

    end_time = time.time()

    print(f"Time to compute percentiles: {end_time - start_time}")
    print(f"Percentiles: {chunked_percentiles}")

    print(f"Combined percentiles: {combined_percentiles}")


if __name__ == "__main__":
    main()
