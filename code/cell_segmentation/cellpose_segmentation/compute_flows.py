import multiprocessing
import os
from time import time
from typing import Callable, Optional, Tuple

import numpy as np
import psutil
import utils
import zarr
from aind_large_scale_prediction._shared.types import ArrayLike, PathLike
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    get_chunk_numbers, get_output_coordinate_overlap, recover_global_position)
from cellpose import core
from cellpose.dynamics import follow_flows
from cellpose.io import logger_setup
from cellpose.models import assign_device
from scipy.ndimage import maximum_filter1d


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def unpad_global_coords(
    global_coord_pos, block_shape, overlap_prediction_chunksize, dataset_shape
):
    unpadded_glob_coord_pos = []
    unpadded_local_coord_pos = []
    for idx, ax_pos in enumerate(global_coord_pos):
        global_curr_left = ax_pos.start + overlap_prediction_chunksize[idx]
        global_curr_right = ax_pos.stop - overlap_prediction_chunksize[idx]

        local_curr_left = overlap_prediction_chunksize[idx]
        local_curr_right = block_shape[idx] - overlap_prediction_chunksize[idx]

        if ax_pos.start == 0:
            # No padding to the left
            global_curr_left = 0
            local_curr_left = 0

        if ax_pos.stop == dataset_shape[idx]:
            global_curr_right = ax_pos.stop
            local_curr_right = block_shape[idx]

        unpadded_glob_coord_pos.append(slice(global_curr_left, global_curr_right))

        unpadded_local_coord_pos.append(slice(local_curr_left, local_curr_right))

    return tuple(unpadded_glob_coord_pos), tuple(unpadded_local_coord_pos)


def computing_overlapping_hist_and_seed_finding(
    p, global_coords, unpadded_local_slice, rpad: int = 20
):

    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = p.shape[0]

    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-0.5 - rpad, shape0[i] + 0.5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    h = h[unpadded_local_slice]
    hmax = hmax[unpadded_local_slice]

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s[:] = s[isort]

    pix = np.array(np.array(seeds).T).astype(np.uint32)
    pix_global = pix + np.array([global_coord.start for global_coord in global_coords])

    return pix_global, pix, h


def large_scale_follow_flows(
    dataset_path: str,
    multiscale: str,
    output_pflow_path: str,
    output_hist_path: str,
    cell_diameter: int,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: Tuple[int, ...],
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
):

    results_folder = os.path.abspath("./results")

    predictions_folder = f"{results_folder}/predictions"
    local_seeds_folder = f"{predictions_folder}/seeds/local_overlap_overlap_unpadded"
    global_seeds_folder = f"{predictions_folder}/seeds/global_overlap_overlap_unpadded"
    hist_seeds_folder = f"{predictions_folder}/hists/hist_overlap_overlap_unpadded"

    create_folder(predictions_folder)
    create_folder(local_seeds_folder)
    create_folder(global_seeds_folder)
    create_folder(hist_seeds_folder)

    co_cpus = 16  # int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder)
    logger.info(f"{20*'='} Z1 Large-Scale Generate Seeds {20*'='}")

    # utils.print_system_information(logger)

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

    ## Creating zarr data loader
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
    overlap_prediction_chunksize = (0,) + tuple(
        [cell_diameter * 2] * len(prediction_chunksize[-3:])
    )
    logger.info(
        f"Overlap size based on cell diameter * 2: {overlap_prediction_chunksize}"
    )

    zarr_data_loader, zarr_dataset = create_data_loader(
        dataset_path=dataset_path,
        multiscale=multiscale,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    # Estimating prediction chunksize overlap
    prediction_chunksize_overlap = np.array(prediction_chunksize) + (
        np.array(overlap_prediction_chunksize) * 2
    )
    logger.info(f"Prediction chunksize overlap: {prediction_chunksize_overlap}")

    output_pflow = zarr.open(
        output_pflow_path,
        "w",
        shape=(3,) + tuple(zarr_dataset.lazy_data.shape[-3:]),
        chunks=(3,) + tuple(prediction_chunksize[-3:]),
        dtype=np.int32,
    )

    output_hist = zarr.open(
        output_hist_path,
        "w",
        shape=tuple(zarr_dataset.lazy_data.shape[-3:]),
        chunks=tuple(prediction_chunksize[-3:]),
        dtype=np.float64,
    )

    logger.info(
        f"Creating zarr flows in path: {output_pflow_path} - {output_pflow} chunks: {output_pflow.chunks}"
    )

    use_GPU = core.use_gpu()
    logger.info(f"GPU activated: {use_GPU}")
    logger_setup()

    sdevice, gpu = assign_device(use_torch=use_GPU, gpu=use_GPU)

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")

    logger.info(f"{20*'='} Starting combination of gradients {20*'='}")
    start_time = time()

    for i, sample in enumerate(zarr_data_loader):

        data_block = np.squeeze(sample.batch_tensor.numpy(), axis=0)

        # Following flows
        pflows, _ = follow_flows(
            data_block / 5.0, niter=200, interp=False, device=sdevice
        )
        pflows = pflows.astype(np.int32)

        (
            global_coord_pos,
            global_coord_positions_start,
            global_coord_positions_end,
        ) = recover_global_position(
            super_chunk_slice=sample.batch_super_chunk[0],
            internal_slices=sample.batch_internal_slice,
        )

        unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
            global_coord_pos=global_coord_pos,
            block_shape=data_block.shape,
            overlap_prediction_chunksize=overlap_prediction_chunksize,
            dataset_shape=zarr_dataset.lazy_data.shape,
        )

        pflows_non_overlaped = pflows[unpadded_local_slice]

        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} Super chunk: {sample.batch_super_chunk} - intern slice: {sample.batch_internal_slice} - global pos: {global_coord_pos}"
        )
        logger.info(
            f"Batch {i}: Saving pflows from {global_coord_pos} to {unpadded_global_slice}. Pshape: {pflows_non_overlaped.shape}"
        )

        logger.info(f"Computing histogram in overlapping chunks")
        # Computing overlapping histogram and pixel seed finding
        global_seeds_overlp, local_seeds_overlp, hist_no_overlp = (
            computing_overlapping_hist_and_seed_finding(
                p=pflows,
                global_coords=unpadded_global_slice[1:],
                unpadded_local_slice=unpadded_local_slice[
                    1:
                ],  # To move the hists to local coord system
                rpad=0,
            )
        )

        output_pflow[unpadded_global_slice] = pflows_non_overlaped
        output_hist[unpadded_global_slice[1:]] = hist_no_overlp

        if local_seeds_overlp.shape[0]:
            logger.info(
                f"Points found in overlapping chunks: {local_seeds_overlp.shape}"
            )
            # Saving seeds
            np.save(
                f"{local_seeds_folder}/local_seeds_{unpadded_global_slice[1:]}.npy",
                local_seeds_overlp,
            )
            np.save(
                f"{global_seeds_folder}/global_seeds_{unpadded_global_slice[1:]}.npy",
                global_seeds_overlp,
            )
            np.save(
                f"{hist_seeds_folder}/local_hist_{unpadded_global_slice[1:]}.npy",
                hist_no_overlp,
            )
            global_seeds_overlp = None
            local_seeds_overlp = None

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
            "cellpose_generate_pflows",
        )


def generate_pflows(combined_gradients_zarr_path, output_pflow, output_hist_path):
    prediction_chunksize = (3, 128, 128, 128)
    super_chunksize = None  # (3, 512, 512, 512)
    target_size_mb = 2048  # None
    n_workers = 0
    batch_size = 1
    cell_diameter = 15

    large_scale_follow_flows(
        dataset_path=combined_gradients_zarr_path,
        output_pflow_path=output_pflow,
        output_hist_path=output_hist_path,
        multiscale=".",
        cell_diameter=cell_diameter,
        prediction_chunksize=prediction_chunksize,
        target_size_mb=target_size_mb,
        n_workers=n_workers,
        batch_size=batch_size,
        super_chunksize=super_chunksize,
    )


if __name__ == "__main__":
    generate_pflows(
        combined_gradients_zarr_path="/Users/camilo.laiton/repositories/dispim-cell-seg-exp/data/good_combined_gradients.zarr",
        output_pflow="./results/pflows.zarr",
        output_hist_path="./results/hists.zarr",
    )
