import multiprocessing
import os
from functools import partial
from time import time
from typing import Callable, Optional, Tuple

import numpy as np
import psutil
import utils
import zarr
from aind_large_scale_prediction._shared.types import ArrayLike, PathLike
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    estimate_output_volume, get_chunk_numbers, get_output_coordinate_overlap,
    get_suggested_cpu_count, recover_global_position)
from aind_large_scale_prediction.io import ImageReaderFactory
from build_seg_masks import large_scale_cellpose_mask_generation
from cellpose import core, models
from cellpose.core import run_3D
from cellpose.dynamics import follow_flows
from cellpose.io import logger_setup
from cellpose.models import Cellpose, CellposeModel, assign_device, transforms
from lazy_deskewing import (create_dispim_config, create_dispim_transform,
                            lazy_deskewing)
from numcodecs import blosc
from scipy.ndimage import (binary_fill_holes, grey_dilation, map_coordinates,
                           maximum_filter1d)


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


def create_zarr_opts(codec: str, compression_level: int) -> dict:
    """
    Creates SmartSPIM options for writing
    the OMEZarr.

    Parameters
    ----------
    codec: str
        Image codec used to write the image

    compression_level: int
        Compression level for the image

    Returns
    -------
    dict
        Dictionary with the blosc compression
        to write the SmartSPIM image
    """
    return {
        "compressor": blosc.Blosc(
            cname=codec, clevel=compression_level, shuffle=blosc.SHUFFLE
        )
    }


def is_point_close_to_boundary(point, shape, max_distance=8):
    """
    Check if a 3D point is close to a boundary by a maximum of max_distance voxels in each dimension.

    Parameters:
        point: tuple or array_like
            The 3D point coordinates (x, y, z).
        shape: tuple or array_like
            The shape of the 3D volume.
        max_distance: int, optional
            The maximum distance to the boundary in each dimension. Default is 8.

    Returns:
        bool
            True if the point is close to a boundary, False otherwise.
    """
    z, y, x = point
    max_z, max_y, max_x = shape

    return (
        x <= max_distance
        or y <= max_distance
        or z <= max_distance
        or max_x - x <= max_distance
        or max_y - y <= max_distance
        or max_z - z <= max_distance
    )


def computing_overlapping_hist_and_seed_finding(
    p, global_coords, distance_from_boundary, rpad: int = 20
):
    """
    Computing overlapping histogram and finding seeds in pixels
    to build the segmentation masks. This function is on Cellpose
    and was modified to accept global coordinates.

    Parameters
    ----------
    p: np.array
        Flows in each axis.

    global_coords: Tuple[slice]
        Global coordinates of the loaded flow

    rpad: int
        Padding width
        Default: 20

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        Global and local pixel location of the seeds to
        build the segmentation masks and the histogram output
    """

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

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s[:] = s[isort]

    pix = np.array(np.array(seeds).T).astype(np.uint32)
    pix_global = pix + np.array([global_coord.start for global_coord in global_coords])

    if pix.shape[0]:
        close_to_boundary = np.apply_along_axis(
            lambda point: is_point_close_to_boundary(
                point, shape0, distance_from_boundary
            ),
            axis=1,
            arr=pix,
        )
        pix = np.concatenate((pix, close_to_boundary[:, None]), axis=1)
        pix_global = np.concatenate((pix_global, close_to_boundary[:, None]), axis=1)

    return pix_global, pix, h


def run_cellpose_net(
    data,
    model,
    compute_masks=False,
    do_3D=True,
    stitch_threshold=0.5,
    flow_threshold=0.5,
    cellprob_threshold=0.0,
    batch_size=8,
    normalize=True,
    min_mask_size=15,
    diameter=15,
):
    data_converted = transforms.convert_image(
        data,
        None,
        channel_axis=None,
        z_axis=0,
        do_3D=(do_3D or stitch_threshold > 0),
        nchan=model.nchan,
    )

    if data_converted.ndim < 4:
        data_converted = data_converted[np.newaxis, ...]

    if diameter is not None and diameter > 0:
        rescale = model.diam_mean / diameter

    elif rescale is None:
        diameter = model.diam_labels
        rescale = model.diam_mean / diameter

    normalize_default = {
        "lowhigh": None,
        "percentile": None,
        "normalize": True,
        "norm3D": do_3D,
        "sharpen_radius": 0,
        "smooth_radius": 0,
        "tile_norm_blocksize": 0,
        "tile_norm_smooth3D": 1,
        "invert": False,
    }

    x = np.asarray(data_converted)
    x = transforms.normalize_img(x, **normalize_default)
    yf, styles = run_3D(
        model.net,
        x,
        rsz=rescale,
        anisotropy=1.0,
        augment=False,
        tile=True,
        tile_overlap=0.1,
    )

    # (dX, dY, dZ)
    dP = np.stack(
        (yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), axis=0
    )
    cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1]

    # masks_cp, styles_cp, dP, cellprob, p = model._run_cp(
    #     data_converted,
    #     compute_masks=compute_masks,
    #     normalize=normalize,
    #     invert=False,
    #     rescale=rescale,
    #     resample=True,
    #     augment=False,
    #     tile=True,
    #     tile_overlap=0.1,
    #     cellprob_threshold=cellprob_threshold,
    #     flow_threshold=flow_threshold,
    #     min_size=min_mask_size,
    #     interp=True,
    #     anisotropy=1.0,
    #     do_3D=do_3D,
    #     stitch_threshold=stitch_threshold
    # )

    return dP, cellprob


def large_scale_combine_gradients(
    dataset_path: str,
    multiscale: str,
    output_combined_gradients_path: str,
    output_cellprob_path: str,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: Tuple[int, ...],
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
):
    """
    Chunked cellpose segmentation

    Parameters
    ----------
    dataset_path: str
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    multiscale: str
        Multiscale to process

    camera: int
        Camera the data was acquired with. It is important
        to apply deskewing upfront.

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize the model will pull from
        the raw data

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

    normalize_image: Optional[bool] = True
        Cellpose parameter to normalize images
        with percentiles. Necessary if the data
        is not normalized in advance

    seg_3d: Optional[bool] = True
        Segmentation for 3D data

    model_name: str = "cyto"
        Model name to use.

    cell_diameter: Optional[int]
        Average cell diameter to segment.
        Default = 15
    """
    results_folder = os.path.abspath("../results")

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder)
    logger.info(f"{20*'='} Z1 Large-Scale Cellpose Segmentation {20*'='}")

    utils.print_system_information(logger)

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
    overlap_prediction_chunksize = (
        0,
        0,
        0,
        0,
        0,
    )  # tuple([cell_diameter*2] * len(prediction_chunksize))
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
        f"Combined gradients: {output_combined_gradients} - chunks: {output_combined_gradients.chunks}"
    )
    logger.info(
        f"Cell probabilities path: {output_cellprob} - chunks: {output_cellprob.chunks}"
    )

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")

    logger.info(f"{20*'='} Starting combination of gradients {20*'='}")
    start_time = time()

    cellprob_threshold = 0.0

    for i, sample in enumerate(zarr_data_loader):
        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"
        )

        # Recover global position of internal chunk
        (
            global_coord_pos,
            global_coord_positions_start,
            global_coord_positions_end,
        ) = recover_global_position(
            super_chunk_slice=sample.batch_super_chunk[0],
            internal_slices=sample.batch_internal_slice,
        )

        data = np.squeeze(sample.batch_tensor.numpy(), axis=0)

        dP = np.stack(
            (
                data[1][0] + data[2][0],  # dZ
                data[0][0] + data[2][1],  # dY
                data[0][1] + data[1][1],  # dX
            ),
            axis=0,
        )

        # Cell probability above threshold
        cell_probability = (
            data[0][-1] + data[1][-1] + data[2][-1] > cellprob_threshold
        ).astype(np.uint8)

        # print("dP shape: ", dP.shape)
        # print("cellprob shape: ", cell_probability.shape)

        # Looking at flows within cell areas
        dP_masked = dP * cell_probability

        # Saving cell probability as binary mask
        cellprob_coord_pos = global_coord_pos[-3:]
        # print("Save cell prob coords: ", cellprob_coord_pos)
        output_cellprob[cellprob_coord_pos] = cell_probability

        # Saving dP
        combined_gradients_coord_pos = (slice(0, 3),) + global_coord_pos[-3:]
        # print("Save combined gradients coords: ", combined_gradients_coord_pos)
        output_combined_gradients[combined_gradients_coord_pos] = dP_masked

        logger.info(
            f"Cell probability coords: {cellprob_coord_pos} - dP masked coords: {combined_gradients_coord_pos}"
        )

    end_time = time()

    # sorted_indices = np.argsort(global_seeds_save[:, 0])
    # sorted_global_seeds_save = global_seeds_save[sorted_indices]
    # sorted_local_seeds_save = local_seeds_save[sorted_indices]
    # sorted_hists =  hist_save[sorted_indices]

    # ids = np.arange(1,sorted_global_seeds_save.shape[0]+1)

    # np.save(f"{results_folder}/sorted_global_seeds.npy", sorted_global_seeds_save)
    # np.save(f"{results_folder}/sorted_local_seeds.npy", sorted_local_seeds_save)
    # np.save(f"{results_folder}/seeds_masks_ids.npy", ids)

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


def combine_gradients(
    gradients_zarr_path, output_combined_gradients_path, output_cellprob_path
):
    prediction_chunksize = (3, 3, 128, 128, 128)
    super_chunksize = (3, 3, 128, 128, 128)
    target_size_mb = 2048  # None
    n_workers = 0
    batch_size = 1

    large_scale_combine_gradients(
        dataset_path=gradients_zarr_path,
        multiscale=".",
        output_combined_gradients_path=output_combined_gradients_path,
        output_cellprob_path=output_cellprob_path,
        prediction_chunksize=prediction_chunksize,
        target_size_mb=target_size_mb,
        n_workers=n_workers,
        batch_size=batch_size,
        super_chunksize=super_chunksize,
    )


if __name__ == "__main__":
    combine_gradients(
        gradients_zarr_path="../results/gradients.zarr",
        output_combined_gradients_path="../results/combined_gradients.zarr",
        output_cellprob_path="../results/combined_cellprob.zarr",
    )
