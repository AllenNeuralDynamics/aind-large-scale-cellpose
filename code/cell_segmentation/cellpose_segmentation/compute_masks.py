import multiprocessing
import os
from glob import glob
from time import time
from typing import Callable, Optional, Tuple

import fastremap
import numpy as np
import psutil
import utils
import zarr
from aind_large_scale_prediction._shared.types import ArrayLike, PathLike
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import recover_global_position
from cellpose import metrics
from scipy.ndimage import binary_fill_holes, grey_dilation, map_coordinates
from skimage.measure import regionprops


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


def create_initial_mask(pflows, local_points, h, cell_ids, rpad=0, iter_points=False):
    expand = np.nonzero(np.ones((3, 3, 3)))
    shape = h.shape
    dims = pflows.shape[0]
    local_points = list(local_points)

    # To get local neighborhood from the expand section
    # to get better accuracy in the mask generation
    if iter_points:
        # 5 iterations
        for iter in range(5):
            # Iterate over each point
            for k in range(len(local_points)):
                # If it's the first iteration I just convert it to list
                if iter == 0:
                    local_points[k] = list(local_points[k])

                # Declare a new voxel
                newpix = []
                # List to check if is inside image block
                iin = []

                # Expand each voxel 3 voxels around it
                for i, e in enumerate(expand):
                    epix = e[:, np.newaxis] + np.expand_dims(local_points[k][i], 0) - 1
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
                igood = h[newpix] > 2
                for i in range(dims):
                    local_points[k][i] = newpix[i][igood]
                if iter == 4:
                    local_points[k] = tuple(local_points[k])

    # Creating seg mask with histogram shape
    mask = np.zeros(np.array(h.shape), np.int32)

    # Planting seeds with estimates global cell ids
    for idx in range(len(local_points)):
        curr_seed = tuple(local_points[idx])
        mask[curr_seed] = cell_ids[idx]

    if rpad > 0:
        # adding padding to the flows if necessary
        for i in range(dims):
            pflows[i] = pflows[i] + rpad

    h = h <= 2
    for iter in range(5):
        mask = grey_dilation(mask, 3)
        mask[h] = 0

    mask = map_coordinates(mask, pflows, mode="nearest")

    return mask


def remove_bad_flow_masks(masks, flows, threshold=0.4, device=None):
    """
    The flows provided here are dP * mask to be optimal.
    This dP is not divided by 5.0 as it is needed to follow
    the flows. flows = dP * cp_mask
    """
    device0 = device
    merrors, _ = metrics.flow_error(masks, flows, device0)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def global_fill_holes_and_remove_small_masks(masks, min_size=15, start_id=0):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )

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


def compute_global_mask(
    pflows,
    local_points,
    hist,
    cell_ids,
    dP_masked=None,
    min_cell_volume=15,
    flow_threshold=2.5,
    rpad=0,
    curr_device=None,
    iter_points=False,
):
    # Creates the initial mask, this one has the
    # previously generated global ids
    initial_mask = create_initial_mask(
        pflows=pflows,
        local_points=local_points,
        h=hist,
        cell_ids=cell_ids,
        rpad=rpad,
        iter_points=iter_points,
    )

    # print("Initial mask shape: ", initial_mask.shape)

    chunked_shape = pflows.shape[1:]
    # Removing relative big masks without renumbering to allow global ids
    uniq, counts = fastremap.unique(initial_mask, return_counts=True)
    big = np.prod(chunked_shape) * 0.4
    bigc = uniq[counts > big]

    second_mask = initial_mask.copy()
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        second_mask = fastremap.mask(second_mask, bigc)

    # unique_values_orig = np.unique(second_mask)

    # if len(unique_values_orig) >= 2: # Make sure that we have something more than 0's
    #     # Camilo Laiton's Note:
    #     # Even though it looks weird to remap the values to old ids
    #     # it's necessary so we don't have skip labels
    #     # but we guarantee that we still have the precomputed global IDs
    #     second_mask_start = unique_values_orig[1]

    #     fastremap.renumber(second_mask, in_place=True, start=second_mask_start)  #convenient to guarantee non-skipped labels
    #     unique_values_new = np.unique(second_mask)

    #     for i, new_id in enumerate(unique_values_new):
    #         second_mask[second_mask == new_id] = unique_values_orig[i]

    #     second_mask = np.reshape(second_mask, chunked_shape)

    # print("Second mask shape: ", second_mask.shape)

    third_mask = second_mask.copy()
    if flow_threshold > 0 and dP_masked is not None:
        # Removing bad flows
        third_mask = remove_bad_flow_masks(
            masks=second_mask,
            flows=dP_masked,  # NOTE: dP_masked is dP*cp_mask stored is the zarr
            threshold=2.5,
            device=curr_device,
        )

    # print("third mask shape: ", third_mask.shape)

    # Fixes the holes in the segmentation mask and removing small
    # masks while keeping the same global IDs
    fourth_mask = global_fill_holes_and_remove_small_masks(
        masks=third_mask.copy(),
        min_size=min_cell_volume,
    )

    # print("fourth mask shape: ", third_mask.shape)

    return fourth_mask


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


def extract_global_to_local(sort_global_ids_with_cells, slice, pad):
    start_pos = []
    stop_pos = []

    for c in slice:
        start_pos.append(c.start - pad)
        stop_pos.append(c.stop + pad)

    start_pos = np.array(start_pos)
    stop_pos = np.array(stop_pos)

    picked_sort_global_ids_with_cells = sort_global_ids_with_cells[
        (sort_global_ids_with_cells[:, 0] >= start_pos[0])
        & (sort_global_ids_with_cells[:, 0] < stop_pos[0])
        & (sort_global_ids_with_cells[:, 1] >= start_pos[1])
        & (sort_global_ids_with_cells[:, 1] < stop_pos[1])
        & (sort_global_ids_with_cells[:, 2] >= start_pos[2])
        & (sort_global_ids_with_cells[:, 2] < stop_pos[2])
    ]

    # print("Picked points: ", picked_sort_global_ids_with_cells, " Start pos: ", start_pos, " pad: ", pad)
    # print("ZYX points: ", picked_sort_global_ids_with_cells[..., :3])
    # print("Localized points: ", picked_sort_global_ids_with_cells[..., :3] - start_pos - pad)
    # print("Real local seeds: ", real_local_seeds)

    # if picked_sort_global_ids_with_cells.shape[0]:
    #     exit()

    picked_sort_global_ids_with_cells[..., :3] = (
        picked_sort_global_ids_with_cells[..., :3] - start_pos - pad
    )

    # Validating seeds are within block boundaries
    picked_sort_global_ids_with_cells = picked_sort_global_ids_with_cells[
        (picked_sort_global_ids_with_cells[:, 0] >= 0)
        & (
            picked_sort_global_ids_with_cells[:, 0]
            <= (stop_pos[0] - start_pos[0]) + pad
        )
        & (picked_sort_global_ids_with_cells[:, 1] >= 0)
        & (
            picked_sort_global_ids_with_cells[:, 1]
            <= (stop_pos[1] - start_pos[1]) + pad
        )
        & (picked_sort_global_ids_with_cells[:, 2] >= 0)
        & (
            picked_sort_global_ids_with_cells[:, 2]
            <= (stop_pos[2] - start_pos[2]) + pad
        )
    ]

    return picked_sort_global_ids_with_cells


def large_scale_generate_masks(
    dataset_path: str,
    multiscale: str,
    hists_path: str,
    seeds_path: str,
    output_seg_mask_path: str,
    cell_diameter: int,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: Tuple[int, ...],
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
):

    global_seeds = None
    if os.path.exists(seeds_path):
        global_regex = f"{seeds_path}/global_seeds_*.npy"
        global_seeds = [np.load(gs) for gs in glob(global_regex)]
        global_seeds = np.concatenate(global_seeds, axis=0)
        indices = np.argsort(global_seeds[:, 0])
        global_seeds = global_seeds[indices]

        n_ids = np.arange(1, 1 + global_seeds.shape[0])
        global_seeds = np.vstack((global_seeds.T, n_ids)).T

    else:
        raise ValueError("Please, provide the global seeds")

    if not global_seeds.shape[0]:
        print("No seeds found, exiting...")
        exit(1)

    results_folder = os.path.abspath("./results")

    co_cpus = 16  # int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder)
    logger.info(f"{20*'='} Z1 Large-Scale Generate Segmentation Mask {20*'='}")

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

    output_seg_masks = zarr.open(
        output_seg_mask_path,
        "w",
        shape=tuple(zarr_dataset.lazy_data.shape[-3:]),
        chunks=tuple(prediction_chunksize[-3:]),
        dtype=np.int32,
    )

    hists = zarr.open(hists_path, "r")

    logger.info(
        f"Creating masks in path: {output_seg_masks} chunks: {output_seg_masks.chunks}"
    )

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")

    logger.info(f"{20*'='} Starting mask generation {20*'='}")
    start_time = time()

    for i, sample in enumerate(zarr_data_loader):

        data_block = np.squeeze(sample.batch_tensor.numpy(), axis=0)

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

        logger.info(
            f"Global slices: {global_coord_pos} - Unpadded global slices: {unpadded_global_slice[1:]} - Local slices: {unpadded_local_slice[1:]}"
        )

        global_points_path = (
            f"{seeds_path}/global_seeds_{unpadded_global_slice[1:]}.npy"
        )

        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} Super chunk: {sample.batch_super_chunk} - intern slice: {sample.batch_internal_slice} - global pos: {global_coord_pos}"
        )

        # Unpadded block mask zeros if seeds don't exist in that area
        chunked_seg_mask = np.zeros(data_block.shape[1:], dtype=np.uint32)

        if os.path.exists(global_points_path):

            picked_gl_to_lc = extract_global_to_local(
                global_seeds.copy(),
                global_coord_pos[1:],
                # unpadded_global_slice[1:],
                0,
            )
            h = hists[global_coord_pos[1:]]
            logger.info(
                f"Computing seg masks in overlapping chunks: {picked_gl_to_lc.shape[0]} - data block shape: {data_block.shape} - hist shape: {h.shape}"
            )

            chunked_seg_mask = compute_global_mask(
                pflows=data_block.astype(np.int32),
                dP_masked=None,
                local_points=picked_gl_to_lc[..., :3],
                hist=h,
                cell_ids=picked_gl_to_lc[..., -1:],
                min_cell_volume=15,
                flow_threshold=0.0,
                rpad=0,
                curr_device=None,
                iter_points=False,
            )

        output_seg_masks[unpadded_global_slice[1:]] = chunked_seg_mask[
            unpadded_local_slice[1:]
        ]

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


def generate_masks(pflows_path, hists_path, seeds_path, output_seg_mask):
    prediction_chunksize = (3, 128, 128, 128)
    super_chunksize = (3, 512, 512, 512)
    target_size_mb = 2048  # None
    cell_diameter = 15
    n_workers = 0
    batch_size = 1

    large_scale_generate_masks(
        dataset_path=pflows_path,
        multiscale=".",
        hists_path=hists_path,
        seeds_path=seeds_path,
        output_seg_mask_path=output_seg_mask,
        cell_diameter=cell_diameter,
        prediction_chunksize=prediction_chunksize,
        target_size_mb=target_size_mb,
        n_workers=n_workers,
        batch_size=batch_size,
        super_chunksize=super_chunksize,
    )


if __name__ == "__main__":
    generate_masks(
        pflows_path="./results/pflows.zarr",
        hists_path="./results/hists.zarr",
        seeds_path="./results/predictions/seeds/global_overlap_overlap_unpadded",
        output_seg_mask="./results/segmentation_mask.zarr",
    )
