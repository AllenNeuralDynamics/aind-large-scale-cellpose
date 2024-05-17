"""
Computes the masks for a large-scale dataset.
We use the computed flows and histograms to be
able to generate the masks. It is necessary that
the flows were computed using overlapping chunks
in each direction.
"""

import logging
import multiprocessing
import os
from glob import glob
from time import time
from typing import Callable, Optional, Tuple, Type

import fastremap
import numpy as np
import psutil
import zarr
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import recover_global_position, unpad_global_coords
from aind_large_scale_prediction.io import ImageReaderFactory
from cellpose import metrics
from scipy.ndimage import binary_fill_holes, grey_dilation, map_coordinates
from skimage.measure import regionprops

from ._shared.types import ArrayLike, PathLike
from .utils import utils


def create_initial_mask(
    pflows: ArrayLike,
    cell_centroids: ArrayLike,
    hist: ArrayLike,
    cell_ids: ArrayLike,
    rpad: Optional[int] = 0,
    iter_points: Optional[bool] = False,
) -> ArrayLike:
    """
    Creates the initial mask for the provided
    flows, cell centroids and histogram for that
    chunk of data.

    Parameters
    ----------
    pflows: ArrayLike
        Flows for the chunk of data. These should be
        overlapping chunks to pick cells with centroids
        out of the current boundaries.

    cell_centroids: ArrayLike
        Cell centroids.

    hist: ArrayLike
        Histogram for the current chunk of data.

    cell_ids: ArrayLike
        Cell ids for each of the cell centroids. These
        are precomputed ids based on the total count of
        centroids.

    rpad: Optional[int]
        Padding that was applied in the computation
        of histograms and centroids.
        Default: 0

    iter_points: Optional[bool]
        If we want to iter each point to expand it in
        a radius of 3 cubic voxels. This helps in the
        detection of weak flows. This increases the
        execution time, but in theory, should resolve
        issues in weak flows.

    Returns
    -------
    ArrayLike
        Current mask for the block of data.
    """
    expand = np.nonzero(np.ones((3, 3, 3)))
    shape = hist.shape
    dims = pflows.shape[0]
    cell_centroids = list(cell_centroids)

    # To get local neighborhood from the expand section
    # to get better accuracy in the mask generation
    if iter_points:
        # 5 iterations
        for iter in range(5):
            # Iterate over each point
            for k in range(len(cell_centroids)):
                # If it's the first iteration I just convert it to list
                if iter == 0:
                    cell_centroids[k] = list(cell_centroids[k])

                # Declare a new voxel
                newpix = []
                # List to check if is inside image block
                iin = []

                # Expand each voxel 3 voxels around it
                for i, e in enumerate(expand):
                    epix = e[:, np.newaxis] + np.expand_dims(cell_centroids[k][i], 0) - 1
                    # Flattenning points around a point inside ZYX
                    epix = epix.flatten()

                    # Check if these generated ZYX points are inside the shape of the block
                    iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                    newpix.append(epix)

                # Check if all pixels are inside within this 3x3x3 radius
                iin = np.all(tuple(iin), axis=0)

                # Leave pixels intact if they are all true
                # Setting this to a new location in memory
                # to avoid problems in overwriting newpix

                pruned_pix = []
                for c_new_pix in newpix:
                    pruned_pix.append(c_new_pix[iin])

                newpix = tuple(pruned_pix)
                # Check for the positions within the histogram
                # to see if it's greater than 2
                # non-sink pixels = True
                igood = hist[newpix] > 2
                for i in range(dims):
                    cell_centroids[k][i] = newpix[i][igood]
                if iter == 4:
                    cell_centroids[k] = tuple(cell_centroids[k])

    # Creating seg mask with histogram shape
    mask = np.zeros(np.array(hist.shape), np.int32)

    # Planting seeds with estimates global cell ids
    for idx in range(len(cell_centroids)):
        curr_seed = tuple(cell_centroids[idx])
        mask[curr_seed] = cell_ids[idx]

    if rpad > 0:
        # adding padding to the flows if necessary
        for i in range(dims):
            pflows[i] = pflows[i] + rpad

    hist = hist <= 2
    for iter in range(5):
        mask = grey_dilation(mask, 3)
        mask[hist] = 0

    mask = map_coordinates(mask, pflows, mode="nearest")

    return mask


def remove_bad_flow_masks(
    masks: ArrayLike,
    flows: ArrayLike,
    threshold: Optional[float] = 0.4,
    device=None,
) -> ArrayLike:
    """
    Removes bad flows within the generated initial mask.
    The flows provided here are dP * mask to be optimal.
    This dP is not divided by 5.0 as it is needed to follow
    the flows. flows = dP * cp_mask

    Parameters
    ----------
    masks: ArrayLike
        Initial computed masks.

    flows: ArrayLike
        Current flows for the computed masks.

    threshold: Optional[float]
        Current threshold for the flows.

    device: Cuda.Device
        Device where the flow error will be computed.
        Default: None

    Returns
    -------
    ArrayLike:
        Mask with pruned segmentations based on error flows.
    """

    device0 = device
    merrors, _ = metrics.flow_error(masks, flows, device0)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def fill_holes_and_remove_small_masks(
    masks: ArrayLike, min_size: Optional[int] = 15, start_id: Optional[int] = 0
) -> ArrayLike:
    """
    Fill holes and removes small segmentation ids
    for possible cells where it's necessary.

    Parameters
    ----------
    masks: ArrayLike
        Segmentation mask for the block of data.

    min_size: Optional[int]
        Minimun size of the mask to be considered
        a cell. Default: 15

    start_id: Optional[int]
        Starting segmentation id, please ignore if
        the cell ids were already precomputed.
        Default: 0

    Returns
    -------
    ArrayLike:
        Array with the pruned segmentation masks.
    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim)

    masks_properties = regionprops(masks)

    for mask_property in masks_properties:
        curr_mask = masks[mask_property.slice] == mask_property.label
        n_pixels = curr_mask.sum()

        # Checking that n_pixels is greater than min size
        if min_size > 0 and n_pixels < min_size:
            masks[mask_property.slice][curr_mask] = 0

        elif n_pixels > 0:
            # Filling holes within current mask
            for k in range(curr_mask.shape[0]):
                curr_mask[k] = binary_fill_holes(curr_mask[k])
            # curr_mask = binary_fill_holes(curr_mask)
            # Relabeling cell to current label after filling holes
            masks[mask_property.slice][curr_mask] = start_id + mask_property.label

    return masks


def compute_chunked_mask(
    pflows: ArrayLike,
    cell_centroids: ArrayLike,
    hist: ArrayLike,
    cell_ids: ArrayLike,
    dP_masked: Optional[ArrayLike] = None,
    min_cell_volume: Optional[float] = 15,
    flow_threshold: Optional[float] = 2.5,
    rpad: Optional[int] = 0,
    curr_device=None,
    iter_points: Optional[bool] = False,
) -> ArrayLike:
    """
    Computes the mask for the current chunk in memory.

    These are the image processing steps applied:
    1. We create an initial mask using the precomputed
    cell ids for the global seeds we obtained from previous steps.
    2. We remove relative big masks without renumbering the global ids.
    3. If flow threshold and dp_masked are provided, we remove bad flows
    by using cellpose.metrics.flow_error function which provides a
    benchmark of the quality of the masks.
    4. We fill holes and remove small masks in the segmentation.

    Returns
    -------
    ArrayLike:
        Segmentation mask with global ids for the chunk of data being processed.
    """
    # Creates the initial mask, this one has the
    # previously generated global ids
    initial_mask = create_initial_mask(
        pflows=pflows,
        cell_centroids=cell_centroids,
        hist=hist,
        cell_ids=cell_ids,
        rpad=rpad,
        iter_points=iter_points,
    )

    chunked_shape = pflows.shape[1:]
    # Removing relative big masks without renumbering to allow global ids
    uniq, counts = fastremap.unique(initial_mask, return_counts=True)
    big = np.prod(chunked_shape) * 0.4
    bigc = uniq[counts > big]

    second_mask = initial_mask.copy()
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        second_mask = fastremap.mask(second_mask, bigc)

    """
    unique_values_orig = np.unique(second_mask)

    if len(unique_values_orig) >= 2: # Make sure that we have something more than 0's
        # Camilo Laiton's Note:
        # Even though it looks weird to remap the values to old ids
        # it's necessary so we don't have skip labels
        # but we guarantee that we still have the precomputed global IDs
        second_mask_start = unique_values_orig[1]

        # convenient to guarantee non-skipped labels
        fastremap.renumber(second_mask, in_place=True, start=second_mask_start)  # noqa: E501
        unique_values_new = np.unique(second_mask)

        for i, new_id in enumerate(unique_values_new):
            second_mask[second_mask == new_id] = unique_values_orig[i]

        second_mask = np.reshape(second_mask, chunked_shape)
    """

    third_mask = second_mask.copy()
    if flow_threshold > 0 and dP_masked is not None:
        # Removing bad flows
        third_mask = remove_bad_flow_masks(
            masks=second_mask,
            flows=dP_masked,  # NOTE: dP_masked is dP*cp_mask stored is the zarr
            threshold=flow_threshold,
            device=curr_device,
        )

    # Fixes the holes in the segmentation mask and removing small
    # masks while keeping the same global IDs
    fourth_mask = fill_holes_and_remove_small_masks(
        masks=third_mask.copy(),
        min_size=min_cell_volume,
    )

    return fourth_mask


def extract_global_to_local(
    global_ids_with_cells: ArrayLike,
    global_slices: Tuple[slice],
    pad: Optional[int] = 0,
) -> ArrayLike:
    """
    Takes global ZYX positions and converts them
    into local ZYX position within the chunk shape.
    It is important to provide the chunk of data with
    overlapping area in each direction to pick cell
    centroids out of the current boundary.

    Parameters
    ----------
    global_ids_with_cells: ArrayLike
        Global ZYX cell centroids with cell ids
        in the last dimension.

    global_slices: Tuple[slice]
        Global coordinate position of this chunk of
        data in the global image.

    pad: Optional[int]
        Padding applied when computing the flows,
        centroids and histograms. Default: 0

    Returns
    -------
    ArrayLike:
        ZYX positions of centroids within the current
        chunk of data.
    """

    start_pos = []
    stop_pos = []

    for c in global_slices:
        start_pos.append(c.start - pad)
        stop_pos.append(c.stop + pad)

    start_pos = np.array(start_pos)
    stop_pos = np.array(stop_pos)

    # Picking locations within current chunk area in global space
    picked_global_ids_with_cells = global_ids_with_cells[
        (global_ids_with_cells[:, 0] >= start_pos[0])
        & (global_ids_with_cells[:, 0] < stop_pos[0])
        & (global_ids_with_cells[:, 1] >= start_pos[1])
        & (global_ids_with_cells[:, 1] < stop_pos[1])
        & (global_ids_with_cells[:, 2] >= start_pos[2])
        & (global_ids_with_cells[:, 2] < stop_pos[2])
    ]

    # Mapping to the local coordinate system of the chunk
    picked_global_ids_with_cells[..., :3] = picked_global_ids_with_cells[..., :3] - start_pos - pad

    # Validating seeds are within block boundaries
    picked_global_ids_with_cells = picked_global_ids_with_cells[
        (picked_global_ids_with_cells[:, 0] >= 0)
        & (picked_global_ids_with_cells[:, 0] <= (stop_pos[0] - start_pos[0]) + pad)
        & (picked_global_ids_with_cells[:, 1] >= 0)
        & (picked_global_ids_with_cells[:, 1] <= (stop_pos[1] - start_pos[1]) + pad)
        & (picked_global_ids_with_cells[:, 2] >= 0)
        & (picked_global_ids_with_cells[:, 2] <= (stop_pos[2] - start_pos[2]) + pad)
    ]

    return picked_global_ids_with_cells


def get_output_seg_data_type(n_cells: int) -> Type:
    """
    Gets the output data type for the segmentation zarr.
    This is important since we need to upscale the masks to
    the high resolution data to be able to detect RNA.

    Parameters
    ----------
    n_cells: int
        Number of detected cells

    Returns
    -------
    Type
        Numpy data type
    """

    max_uint_16 = np.iinfo(np.uint16).max + 1
    max_uint_32 = np.iinfo(np.uint32).max + 1
    max_uint_64 = np.iinfo(np.uint64).max + 1

    selected_dtype = None

    # Cell id range 0 - 65535
    if n_cells < max_uint_16:
        selected_dtype = np.uint16

    # Cell id range 0 - 4294967295
    elif n_cells < max_uint_32:
        selected_dtype = np.uint32

    # Cell id range 0 - 18446744073709551615
    elif n_cells < max_uint_64:
        selected_dtype = np.uint64

    else:
        raise NotImplementedError("Data type not implemented!")

    return selected_dtype


def execute_worker(
    data: ArrayLike,
    batch_super_chunk: Tuple[slice],
    batch_internal_slice: Tuple[slice],
    overlap_prediction_chunksize: Tuple[int],
    dataset_shape: Tuple[int],
    cell_centroids_path: PathLike,
    output_seg_masks: zarr.core.Array,
    original_dataset_shape: Tuple[int],
    global_seeds: PathLike,
    hists: ArrayLike,
    min_cell_volume: int,
    flow_threshold: float,
    logger: logging.Logger,
):
    """
    Function that executes each worker. It takes
    the flows, histograms and global seeds (cell centroids)
    to compute the segmentation masks.

    Parameters
    ----------
    data: ArrayLike
        Data to process.

    batch_super_chunk: Tuple[slice]
        Slices of the super chunk loaded in shared memory.

    batch_internal_slice: Tuple[slice]
        Internal slice of the current chunk of data. This
        is a local coordinate system based on the super chunk.

    overlap_prediction_chunksize: Tuple[int]
        Overlap area between chunks.

    dataset_shape: Tuple[int]
        Entire dataset shape.

    cell_centroids_path: PathLike
        Path where the cell centroids are stored.

    output_seg_masks: zarr.core.Array
        Zarr where the segmentation mask will be written.

    original_dataset_shape: Tuple[int]
        Entire dataset shape.

    global_seeds: PathLike
        Path where the global seeds are stored.

    hists: ArrayLike
        lazy zarr dataset with histograms.

    min_cell_volume: int
        Minimum cell volume.

    flow_threshold: float
        Flow threshold.

    logger: logging.Logger
        Logging object.
    """

    data = np.squeeze(data, axis=0)

    (
        global_coord_pos,
        global_coord_positions_start,
        global_coord_positions_end,
    ) = recover_global_position(
        super_chunk_slice=batch_super_chunk,  # sample.batch_super_chunk[0],
        internal_slices=batch_internal_slice,  # sample.batch_internal_slice,
    )

    unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
        global_coord_pos=global_coord_pos,
        block_shape=data.shape,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        dataset_shape=dataset_shape,  # zarr_dataset.lazy_data.shape,
    )

    logger.info(
        f"Global slices: {global_coord_pos} - Unpadded global slices: {unpadded_global_slice[1:]} - Local slices: {unpadded_local_slice[1:]}"  # noqa: E501
    )

    global_points_path = f"{cell_centroids_path}/global_seeds_{unpadded_global_slice[1:]}.npy"

    # Unpadded block mask zeros if seeds don't exist in that area
    chunked_seg_mask = np.zeros(data.shape[1:], dtype=np.uint32)

    if os.path.exists(global_points_path):

        picked_gl_to_lc = extract_global_to_local(
            global_seeds.copy(),
            global_coord_pos[1:],
            0,
        )
        chunked_hist = hists[global_coord_pos[1:]]
        logger.info(
            f"Worker [{os.getpid()}] Computing seg masks in overlapping chunks: {picked_gl_to_lc.shape[0]} - data block shape: {data.shape} - hist shape: {chunked_hist.shape}"  # noqa: E501
        )

        chunked_seg_mask = compute_chunked_mask(
            pflows=data.astype(np.int32),
            dP_masked=None,
            cell_centroids=picked_gl_to_lc[..., :3],
            hist=chunked_hist,
            cell_ids=picked_gl_to_lc[..., -1:],
            min_cell_volume=min_cell_volume,
            flow_threshold=flow_threshold,
            rpad=0,
            curr_device=None,
            iter_points=False,
        )

    # Adding new axis for the output segmentation
    unpad_chunked_seg_mask = chunked_seg_mask[unpadded_local_slice[1:]]
    unpad_chunked_seg_mask = utils.pad_array_n_d(
        arr=unpad_chunked_seg_mask, dim=len(original_dataset_shape)
    )

    output_slices = (
        slice(0, 1),
        slice(0, 1),
    ) + unpadded_global_slice[1:]
    output_chunk_shape = output_seg_masks[output_slices].shape

    unpad_chunked_seg_mask = unpad_chunked_seg_mask[
        :,
        :,
        : output_chunk_shape[-3],
        : output_chunk_shape[-2],
        : output_chunk_shape[-1],
    ]

    output_seg_masks[output_slices] = unpad_chunked_seg_mask


def _execute_worker(params):
    """
    Worker interface to provide parameters
    """
    execute_worker(**params)


def generate_masks(
    dataset_path: PathLike,
    multiscale: str,
    hists_path: PathLike,
    cell_centroids_path: PathLike,
    output_seg_mask_path: PathLike,
    original_dataset_shape: Tuple[int, ...],
    axis_overlap: int,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    results_folder: PathLike,
    super_chunksize: Tuple[int, ...],
    min_cell_volume: Optional[int] = 80,
    flow_threshold: Optional[float] = 0.0,
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
):
    """
    Computes the global segmentation masks.

    Parameters
    ----------
    dataset_path: str
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    multiscale: str
        Multiscale to process

    hists_path: PathLike
        Path where the histograms are stored.

    cell_centroids_path: PathLike
        Path where the global cell centroids are stored.

    output_seg_mask_path: PathLike
        Path where we want to output the global
        segmentation mask.

    axis_overlap: int
        Overlap in each axis. This would be 2*axis_overlap
        since it will be in each side. Recommended to be
        cell_diameter * 2.

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

    results_folder: PathLike
        Path where the results folder for cell segmentation
        is located.

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size that will be in memory at a
        time from the raw data. If provided, then
        target_size_mb is ignored. Default: None

    min_cell_volume: Optional[int]
        Minimum cell volume for prunning masks
        Default: 0 (no prunning)

    flow_threshold: Optional[float]
        Flow threshold for prunning bad flows. Currently,
        it is being ignored, it needs the dp_masked data.

    """
    axis_overlap = np.ceil(axis_overlap).astype(np.uint16)

    global_seeds = None
    output_seg_dtype = None
    if os.path.exists(cell_centroids_path):
        global_regex = f"{cell_centroids_path}/global_seeds_*.npy"
        global_seeds = [np.load(gs) for gs in glob(global_regex)]
        global_seeds = np.concatenate(global_seeds, axis=0)
        indices = np.argsort(global_seeds[:, 0])
        global_seeds = global_seeds[indices]

        n_ids = np.arange(1, 1 + global_seeds.shape[0])

        output_seg_dtype = np.uint32  # get_output_seg_data_type(n_cells=n_ids.shape[0])
        global_seeds = np.vstack((global_seeds.T, n_ids)).T
        np.save(f"{results_folder}/cell_centroids.npy", global_seeds)

    else:
        raise ValueError("Please, provide the global seeds")

    if not global_seeds.shape[0]:
        print("No seeds found, exiting...")
        exit(1)  # EXIT_FAILURE

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder, mode="a")
    logger.info(f"{20*'='} Large-Scale Cellpose - Generate Masks {20*'='}")

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
    overlap_prediction_chunksize = (0,) + tuple([axis_overlap * 2] * len(prediction_chunksize[-3:]))
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

    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
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

    original_dataset_shape = (
        1,
        1,
    ) + tuple(original_dataset_shape[-3:])
    output_chunk_size = (
        1,
        1,
    ) + tuple(prediction_chunksize[-3:])
    logger.info(f"Prediction chunksize overlap: {prediction_chunksize_overlap}")
    logger.info(
        f"Original dataset shape: {original_dataset_shape} - output chunksize: {output_chunk_size}"
    )

    output_seg_masks = zarr.open(
        output_seg_mask_path,
        "w",
        shape=original_dataset_shape,  # tuple(zarr_dataset.lazy_data.shape[-3:]),
        chunks=output_chunk_size,
        dtype=output_seg_dtype,
    )

    hists = zarr.open(hists_path, "r")

    logger.info(f"Creating masks in path: {output_seg_masks} chunks: {output_seg_masks.chunks}")

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches} - Samples per iteration: {samples_per_iter}")

    logger.info(f"{20*'='} Starting mask generation {20*'='}")
    start_time = time()

    # Setting exec workers to CO CPUs
    exec_n_workers = co_cpus

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=exec_n_workers)

    # Variables for multiprocessing
    picked_blocks = []
    curr_picked_blocks = 0

    logger.info(f"Number of workers processing data: {exec_n_workers}")

    for i, sample in enumerate(zarr_data_loader):

        picked_blocks.append(
            {
                "data": sample.batch_tensor.numpy(),
                "batch_super_chunk": sample.batch_super_chunk[0],
                "batch_internal_slice": sample.batch_internal_slice,
                "overlap_prediction_chunksize": overlap_prediction_chunksize,
                "dataset_shape": zarr_dataset.lazy_data.shape,
                "cell_centroids_path": cell_centroids_path,
                "output_seg_masks": output_seg_masks,
                "original_dataset_shape": original_dataset_shape,
                "global_seeds": global_seeds,
                "min_cell_volume": min_cell_volume,
                "flow_threshold": flow_threshold,
                "hists": hists,
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

            logger.info(
                f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs -> Batch {i} Last slice in list: {sample.batch_internal_slice_global}"  # noqa: E501
            )

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
            "cellpose_generate_masks",
        )


def main(
    pflows_path: PathLike,
    hists_path: PathLike,
    cell_centroids_path: PathLike,
    output_seg_mask: PathLike,
):
    """
    Function to generate the masks.

    Parameters
    ----------
    pflows_path: PathLike
        Path where the flows are stored.

    hists_path: PathLike
        Path where the histograms were stored.

    cell_centroids_path: PathLike
        Path where the global cell centroids are
        stored. These are numpy arrays.

    output_seg_mask: PathLike
        Path where we want to store the segmentation
        masks for the current dataset.
    """
    prediction_chunksize = (3, 128, 128, 128)
    super_chunksize = (3, 512, 512, 512)
    target_size_mb = 2048  # None
    cell_diameter = 15
    n_workers = 0
    batch_size = 1

    generate_masks(
        dataset_path=pflows_path,
        multiscale=".",
        hists_path=hists_path,
        cell_centroids_path=cell_centroids_path,
        output_seg_mask_path=output_seg_mask,
        original_dataset_shape=(1, 1, 114, 827, 598),
        axis_overlap=cell_diameter,
        prediction_chunksize=prediction_chunksize,
        target_size_mb=target_size_mb,
        n_workers=n_workers,
        batch_size=batch_size,
        super_chunksize=super_chunksize,
        min_cell_volume=0,
        flow_threshold=0.0,
    )


if __name__ == "__main__":
    main(
        pflows_path="../../results/pflows.zarr",
        hists_path="../../results/hists.zarr",
        seeds_path="../../results/flow_results/seeds/global",
        output_seg_mask="../../results/segmentation_mask.zarr",
    )
