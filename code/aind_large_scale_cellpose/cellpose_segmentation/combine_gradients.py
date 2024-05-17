"""
3D combination of gradients that were previously
predicted. This is a local operation that happens
that does not require to run in overlapping chunks.
"""

import logging
import multiprocessing
import os
from time import time
from typing import Callable, Optional, Tuple

import numpy as np
import psutil
import zarr
from aind_large_scale_prediction._shared.types import ArrayLike, PathLike
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import recover_global_position
from aind_large_scale_prediction.io import ImageReaderFactory

from .utils import utils


def execute_worker(
    data: ArrayLike,
    batch_super_chunk: Tuple[slice],
    batch_internal_slice: Tuple[slice],
    cellprob_threshold: float,
    output_cellprob: zarr.core.Array,
    output_combined_gradients: zarr.core.Array,
    logger: logging.Logger,
):
    """
    Function that executes each worker. It takes
    the predicted gradients, combines and generates
    the cell probability zarr and combined gradient zarr.

    Parameters
    ----------
    data: ArrayLike
        Data to process.

    batch_super_chunk: Tuple[slice]
        Slices of the super chunk loaded in shared memory.

    batch_internal_slice: Tuple[slice]
        Internal slice of the current chunk of data. This
        is a local coordinate system based on the super chunk.

    cellprob_threshold: float
        Cell probability threshold.

    output_cellprob: zarr.core.Array
        Zarr dataset where we write the cell probabilities.

    output_combined_gradients: zarr.core.Array
        Zarr dataset where we write the combined gradients.

    logger: logging.Logger
        Logging object
    """
    # Recover global position of internal chunk
    (
        global_coord_pos,
        global_coord_positions_start,
        global_coord_positions_end,
    ) = recover_global_position(
        super_chunk_slice=batch_super_chunk,  # sample.batch_super_chunk[0],
        internal_slices=batch_internal_slice,  # sample.batch_internal_slice,
    )

    data = np.squeeze(data, axis=0)  # sample.batch_tensor.numpy()

    dP = np.stack(
        (
            data[1][0] + data[2][0],  # dZ
            data[0][0] + data[2][1],  # dY
            data[0][1] + data[1][1],  # dX
        ),
        axis=0,
    )

    # Cell probability above threshold
    cell_probability = (data[0][-1] + data[1][-1] + data[2][-1] > cellprob_threshold).astype(
        np.uint8
    )

    # Looking at flows within cell areas
    dP_masked = dP * cell_probability

    # Saving cell probability as binary mask
    cellprob_coord_pos = global_coord_pos[-3:]
    output_cellprob[cellprob_coord_pos] = cell_probability

    # Saving dP
    combined_gradients_coord_pos = (slice(0, 3),) + global_coord_pos[-3:]
    output_combined_gradients[combined_gradients_coord_pos] = dP_masked

    logger.info(
        f"Worker [{os.getpid()}] - Cell probability coords: {cellprob_coord_pos} - dP masked coords: {combined_gradients_coord_pos}"  # noqa: E501
    )


def _execute_worker(params):
    """
    Worker interface to provide parameters
    """
    execute_worker(**params)


def combine_gradients(
    dataset_path: PathLike,
    multiscale: str,
    output_combined_gradients_path: PathLike,
    output_cellprob_path: PathLike,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: Tuple[int, ...],
    results_folder: PathLike,
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
):
    """
    Local 3D combination of predicted gradients.
    This operation is necessary before following
    the flows to the centers identified cells.

    Parameters
    ----------
    dataset_path: str
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    multiscale: str
        Multiscale to process

    output_combined_gradients_path: PathLike
        Path where we want to output the combined gradients.

    output_cellprob_path: PathLike
        Path where we want to output the cell proabability
        maps. It is not completely necessary to save them
        but it is good for quality control.

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize.

    target_size_mb: int
        Target size in megabytes the data loader will
        load in memory at a time

    n_workers: int
        Number of workers that will concurrently pull
        data from the shared super chunk in memory

    batch_size: int
        Batch size

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size that will be in memory at a
        time from the raw data. If provided, then
        target_size_mb is ignored. Default: None

    results_folder: PathLike
        Path where the results folder for cell segmentation
        is located.

    """

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder, mode="a")
    logger.info(f"{20*'='} Large-Scale Cellpose - Combination of Gradients {20*'='}")

    logger.info(f"Processing dataset {dataset_path}")

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    # Creating zarr data loader
    logger.info("Creating chunked data loader")
    shm_memory = psutil.virtual_memory()
    logger.info(f"Shared memory information: {shm_memory}")

    # The device we will use and pinning memory to speed things up
    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    # Getting overlap prediction chunksize
    overlap_prediction_chunksize = (
        0,
        0,
        0,
        0,
        0,
    )
    logger.info(f"Overlap size based on cell diameter * 2: {overlap_prediction_chunksize}")

    lazy_data = (
        ImageReaderFactory()
        .create(
            data_path=dataset_path,
            parse_path=False,
            multiscale=multiscale,
        )
        .as_dask_array()
    )

    # Creation of zarr data loader
    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=lazy_callback_fn,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    logger.info(f"Creating zarr gradients in path: {output_combined_gradients_path}")
    output_combined_gradients = zarr.open(
        output_combined_gradients_path,
        "w",
        shape=(3,) + zarr_dataset.lazy_data.shape[-3:],  # dZ, dY, dX
        chunks=(1,) + tuple(prediction_chunksize[-3:]),
        dtype=np.float32,
    )

    output_cellprob = zarr.open(
        output_cellprob_path,
        "w",
        shape=zarr_dataset.lazy_data.shape[-3:],
        chunks=tuple(prediction_chunksize[-3:]),
        dtype=np.uint8,
    )
    logger.info(
        f"Combined gradients: {output_combined_gradients} - chunks: {output_combined_gradients.chunks}"  # noqa: E501
    )
    logger.info(f"Cell probabilities path: {output_cellprob} - chunks: {output_cellprob.chunks}")

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches} - Samples per iteration: {samples_per_iter}")

    logger.info(f"{20*'='} Starting combination of gradients {20*'='}")
    start_time = time()

    cellprob_threshold = 0.0

    # Setting exec workers to CO CPUs
    exec_n_workers = co_cpus

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=exec_n_workers)

    # Variables for multiprocessing
    picked_blocks = []
    curr_picked_blocks = 0

    logger.info(f"Number of workers processing data: {exec_n_workers}")

    for i, sample in enumerate(zarr_data_loader):
        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"  # noqa: E501
        )

        picked_blocks.append(
            {
                "data": sample.batch_tensor.numpy(),
                "batch_super_chunk": sample.batch_super_chunk[0],
                "batch_internal_slice": sample.batch_internal_slice,
                "cellprob_threshold": cellprob_threshold,
                "output_cellprob": output_cellprob,
                "output_combined_gradients": output_combined_gradients,
                "logger": logger,
            }
        )
        curr_picked_blocks += 1

        if curr_picked_blocks == exec_n_workers:

            # Assigning blocks to execution workers
            jobs = [
                pool.apply_async(_execute_worker, args=(picked_block,))
                for picked_block in picked_blocks
            ]

            logger.info(f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs")

            # Wait for all processes to finish
            results = [job.get() for job in jobs]  # noqa: F841

            # Setting variables back to init
            curr_picked_blocks = 0
            picked_blocks = []

    if curr_picked_blocks != 0:
        logger.info(f"Blocks not processed inside of loop: {curr_picked_blocks}")
        # Assigning blocks to execution workers
        jobs = [
            pool.apply_async(_execute_worker, args=(picked_block,))
            for picked_block in picked_blocks
        ]

        # Wait for all processes to finish
        results = [job.get() for job in jobs]  # noqa: F841

        # Setting variables back to init
        curr_picked_blocks = 0
        picked_blocks = []

    # Closing pool of workers
    pool.close()

    end_time = time()

    logger.info(f"Processing time: {end_time - start_time} seconds")

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            results_folder,
            "cellpose_combine_gradients",
        )


def main():
    """Main function"""
    combine_gradients(
        dataset_path="../results/gradients.zarr",
        multiscale=".",
        output_combined_gradients_path="../results/combined_gradients.zarr",
        output_cellprob_path="../results/combined_cellprob.zarr",
        prediction_chunksize=(3, 3, 128, 128, 128),
        super_chunksize=(3, 3, 128, 128, 128),
        target_size_mb=2048,
        n_workers=0,
        batch_size=1,
    )


if __name__ == "__main__":
    main()
