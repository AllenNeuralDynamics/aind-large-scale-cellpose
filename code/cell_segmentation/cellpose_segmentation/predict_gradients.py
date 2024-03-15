import multiprocessing
import os
from time import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import psutil
import utils
import zarr
from aind_large_scale_prediction._shared.types import ArrayLike, PathLike
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import recover_global_position
from cellpose import core
from cellpose.core import run_net
from cellpose.io import logger_setup
from cellpose.models import CellposeModel, assign_device, transforms
from numcodecs import blosc
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


def run_2D_cellpose(
    net,
    imgs,
    p,
    batch_size=8,
    rsz=1.0,
    anisotropy=None,
    augment=False,
    tile=True,
    tile_overlap=0.1,
    bsize=224,
    progress=None,
):
    sstr = ["YX", "ZY", "ZX"]
    # print("anisotropy: ", anisotropy, " - rsz: ", rsz)
    if anisotropy is not None:
        rescaling = [[rsz, rsz], [rsz * anisotropy, rsz], [rsz * anisotropy, rsz]]
    else:
        rescaling = [rsz] * 3

    # print("rescaling: ", rescaling)

    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(3, 0, 1, 2), (3, 1, 0, 2), (3, 1, 2, 0)]
    nout = net.nout
    # print("images shape: ", imgs.shape, pm[p], " - rescaling: ", rescaling, rescaling[p])

    xsl = imgs.copy().transpose(pm[p])
    # rescale image for flow computation
    shape = xsl.shape
    xsl = transforms.resize_image(xsl, rsz=rescaling[p])
    # per image
    print(
        "running %s: %d planes of size (%d, %d)"
        % (sstr[p], shape[0], shape[1], shape[2])
    )
    y, style = run_net(
        net,
        xsl,
        batch_size=batch_size,
        augment=augment,
        tile=tile,
        bsize=bsize,
        tile_overlap=tile_overlap,
    )
    y = transforms.resize_image(y, shape[1], shape[2])
    y = y.transpose(ipm[p])
    if progress is not None:
        progress.setValue(25 + 15 * p)
    return y, style


def run_cellpose_net(
    data,
    model,
    axis,
    compute_masks=False,
    batch_size=8,
    normalize=True,
    diameter=15,
    rsz=1.0,
    anisotropy=1.0,
):
    data_converted = transforms.convert_image(
        data, None, channel_axis=None, z_axis=0, do_3D=True, nchan=model.nchan
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
        "norm3D": False,
        "sharpen_radius": 0,
        "smooth_radius": 0,
        "tile_norm_blocksize": 0,
        "tile_norm_smooth3D": 1,
        "invert": False,
    }

    x = np.asarray(data_converted)
    x = transforms.normalize_img(x, **normalize_default)

    # print("Data shape: ", x.shape, " rescale: ", rescale)

    y, style = run_2D_cellpose(
        model.net,
        x,
        p=axis,
        rsz=rescale,
        anisotropy=anisotropy,
        augment=False,
        tile=True,
        tile_overlap=0.1,
    )

    return y


def large_scale_cellpose_gradients(
    dataset_path: str,
    multiscale: str,
    output_gradients_path: PathLike,
    axis: int,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: Tuple[int, ...],
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    normalize_image: Optional[bool] = True,
    seg_3d: Optional[bool] = True,
    model_name: Optional[str] = "cyto",
    cell_diameter: Optional[int] = 15,
):
    results_folder = os.path.abspath("../results")

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder)
    logger.info(f"{20*'='} Z1 Large-Scale Cellpose Segmentation {20*'='}")

    utils.print_system_information(logger)

    logger.info(f"Processing dataset {dataset_path} with mulsticale {multiscale}")

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

    shape = None
    if axis == 0:
        logger.info(f"Creating zarr gradients in path: {output_gradients_path}")
        output_gradients = zarr.open(
            output_gradients_path,
            "w",
            shape=(
                3,
                3,
            )
            + zarr_dataset.lazy_data.shape,
            chunks=(
                1,
                3,
            )
            + tuple(prediction_chunksize),
            dtype=np.float32,
        )
        shape = zarr_dataset.lazy_data.shape

    else:
        # Reading back output gradients
        output_gradients = zarr.open(
            output_gradients_path,
            "a",
        )
        shape = output_gradients.shape

    logger.info(
        f"Gradients: {output_gradients} chunks: {output_gradients.chunks} - Current shape: {shape}"
    )

    # Setting up cellpose
    use_GPU = core.use_gpu()
    logger.info(f"GPU activated: {use_GPU}")
    logger_setup()

    sdevice, gpu = assign_device(use_torch=use_GPU, gpu=use_GPU)
    model = CellposeModel(
        gpu=gpu, model_type=model_name, diam_mean=cell_diameter, device=sdevice
    )

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")

    logger.info(
        f"{20*'='} Starting estimation of cellpose combined gradients - Axis {axis} {20*'='}"
    )
    start_time = time()

    for i, sample in enumerate(zarr_data_loader):
        data = sample.batch_tensor.numpy()[0, ...]

        if data.shape != prediction_chunksize:
            logger.info(
                f"Non-uniform block of data... {data.shape} - {prediction_chunksize}"
            )
            continue

        # Recover global position of internal chunk
        (
            global_coord_pos,
            global_coord_positions_start,
            global_coord_positions_end,
        ) = recover_global_position(
            super_chunk_slice=sample.batch_super_chunk[0],
            internal_slices=sample.batch_internal_slice,
        )

        global_coord_pos = (slice(axis, axis + 1), slice(0, 3)) + global_coord_pos

        # # Estimating plane gradient
        y = run_cellpose_net(
            data=data,
            model=model,
            axis=axis,
            compute_masks=False,
            batch_size=8,
            normalize=True,
            diameter=15,
            rsz=1.0,
            anisotropy=1.0,
        )

        global_coord_pos = list(global_coord_pos)

        if shape[-1] < global_coord_pos[-1].stop:
            global_coord_pos[-1] = slice(global_coord_pos[-1].start, shape[-1])

        if shape[-2] < global_coord_pos[-2].stop:
            global_coord_pos[-2] = slice(global_coord_pos[-2].start, shape[-2])

        if shape[-3] < global_coord_pos[-3].stop:
            global_coord_pos[-3] = slice(global_coord_pos[-3].start, shape[-3])

        global_coord_pos = tuple(global_coord_pos)
        logger.info(f"Writing to: {global_coord_pos}")

        output_gradients[global_coord_pos] = np.expand_dims(y, axis=0)

        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device} - global_coords: {global_coord_pos} - Pred shape: {y.shape}"
        )

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
            "cellpose_segmentation",
        )


def main():
    # Data loader params
    BUCKET_NAME = "aind-open-data"
    IMAGE_PATH = "HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49"
    TILE_NAME = "channel_405.zarr"
    # dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    dataset_path = f"/data/{IMAGE_PATH}/{TILE_NAME}"
    multiscale = "2"

    image_shape = zarr.open(f"{dataset_path}/{multiscale}", "r").shape

    super_chunksize = None  # (384, 768, 768)
    target_size_mb = 3072  # None
    n_workers = 16
    batch_size = 1

    # Cellpose params
    model_name = "cyto"
    seg_3d = True
    normalize_image = True
    cell_diameter = 15  # 8 #

    # output gradients
    output_gradients_path = "../results/gradients.zarr"
    axes_names = ["YX", "ZX", "ZY"]

    slices_per_axis = [40, 80, 80]

    for axis in range(2, 3):

        slice_per_axis = slices_per_axis[axis]
        prediction_chunksize = None
        if axis == 0:
            prediction_chunksize = (slice_per_axis, image_shape[-2], image_shape[-1])

        elif axis == 1:
            prediction_chunksize = (image_shape[-3], slice_per_axis, image_shape[-1])

        elif axis == 2:
            prediction_chunksize = (image_shape[-3], image_shape[-2], slice_per_axis)

        print(
            f"{20*'='} Large-scale computation of gradients in {axes_names[axis]} - Prediction chunksize {prediction_chunksize} {20*'='}"
        )

        large_scale_cellpose_gradients(
            dataset_path=dataset_path,
            multiscale=multiscale,
            output_gradients_path=output_gradients_path,
            axis=axis,
            prediction_chunksize=prediction_chunksize,
            target_size_mb=target_size_mb,
            n_workers=n_workers,
            batch_size=batch_size,
            super_chunksize=super_chunksize,
            normalize_image=normalize_image,
            seg_3d=seg_3d,
            model_name=model_name,
            cell_diameter=cell_diameter,
        )


if __name__ == "__main__":
    main()
