"""
Large-scale prediction of gradients. We are
computing gradients in entire 2D planes which
are ZY, ZX and XY.
"""

import multiprocessing
from time import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
import zarr
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data, recover_global_position)
from cellpose.core import run_net, use_gpu
from cellpose.io import logger_setup
from cellpose.models import CellposeModel, assign_device, transforms

from ._shared.types import ArrayLike, PathLike
from .compute_percentiles import compute_percentiles
from .utils import utils


def run_2D_cellpose(
    net: CellposeModel,
    imgs: ArrayLike,
    img_axis: int,
    batch_size: Optional[int] = 8,
    rsz: Optional[float] = 1.0,
    anisotropy: Optional[float] = None,
    augment: Optional[bool] = False,
    tile: Optional[bool] = True,
    tile_overlap: Optional[float] = 0.1,
    bsize: Optional[int] = 224,
    progress=None,
) -> Tuple[ArrayLike]:
    """
    Runs cellpose on 2D images.

    Parameters
    ----------
    net: CellposeModel
        Initialized cellpose model to run
        inference on.

    imgs: ArrayLike
        Images to run inference on.

    img_axis: int
        Integer pointing to the image axis
        these 2D images belong to. Our images
        are in ZYX order.

    batch_size: Optional[int]
        Batch size. Default: 8

    rsz: Optional[float] = 1.0
        Rescaling factor in each dimension.
        Default: 1.0

    anisotropy: Optional[float] = None
        Anisotropy between orientations.
        Default: None

    augment: Optional[bool] = False
        tiles image with overlapping tiles and flips overlapped regions to augment.
        Default: False.

    tile: Optional[bool] = True
        tiles image to ensure GPU/CPU memory usage limited (recommended).
        Default: True.

    tile_overlap: Optional[float] = 0.1
        Fraction of overlap of tiles when computing flows.
        Default: 0.1.

    bsize: Optional[int] = 224
        block size for tiles, recommended to keep at 224, like in training.
        Default: 224.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Predicted gradients and style.
    """
    sstr = ["XY", "ZX", "ZY"]
    if anisotropy is not None:
        rescaling = [[rsz, rsz], [rsz * anisotropy, rsz], [rsz * anisotropy, rsz]]
    else:
        rescaling = [rsz] * 3

    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(3, 0, 1, 2), (3, 1, 0, 2), (3, 1, 2, 0)]
    xsl = imgs.copy().transpose(pm[img_axis])

    shape = xsl.shape
    xsl = transforms.resize_image(xsl, rsz=rescaling[img_axis])

    print(
        "running %s: %d planes of size (%d, %d)"
        % (sstr[img_axis], shape[0], shape[1], shape[2])
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
    y = y.transpose(ipm[img_axis])

    if progress is not None:
        progress.setValue(25 + 15 * img_axis)

    return y, style


def percentile_normalization(
    data: ArrayLike, chn_percentiles: Dict, channels: List[int]
) -> ArrayLike:
    """
    Performs percentile normalization based on
    global computed percentiles.

    Parameters
    ----------
    data: ArrayLike
        Data to be normalized.

    chn_percentiles: Dict
        Dictionary with the percentiles.

    channels: List[int, int]
        Channel ids to compute the percentile
        normalization on.

    Returns
    -------
    ArrayLike
        Normalized data
    """

    norm_data = data.copy().astype(np.float32)

    # After converting the image, the channels are at the end

    # Correcting the channels based on channel percentiles
    for chn_idx in channels:
        curr_percentiles = chn_percentiles[chn_idx]
        norm_data[..., chn_idx] = np.maximum(
            norm_data[..., chn_idx], curr_percentiles[0]
        )
        norm_data[..., chn_idx] = (norm_data[..., chn_idx] - curr_percentiles[0]) / (
            curr_percentiles[1] - curr_percentiles[0]
        )

    return norm_data


def run_cellpose_net(
    data: ArrayLike,
    model: CellposeModel,
    axis: int,
    channels: Optional[List[int]] = [0, 0],
    z_axis: Optional[int] = 0,
    normalize: Optional[bool] = False,
    diameter: Optional[int] = 15,
    anisotropy: Optional[float] = 1.0,
    channel_percentiles: Optional[Dict] = None,
) -> ArrayLike:
    """
    Runs cellpose in stacks of 2D images.

    Parameters
    ----------
    data: ArrayLike
        Stack of 2D images to be processed.

    model: CellposeModel
        Initialized cellpose model.

    axis: int
        Image axis to be processed.

    normalize: Optional[bool]
        If we want to normalize the data
        using percentile normalization.
        If False, please provide percentiles for
        the channels. These can be globally computed.
        Default: False

    diameter: Optional[int]
        Mean cell diameter
        Default: 15

    anisotropy: Optional[float]
        Anisotropy factor

    channel_percentiles: Optional[List[float, float]]
        Precomputed channel percentiles. This is a list
        that contains the min and max values of the entire dataset
        based on the chunked percentile estimation.

    Returns
    -------
    ArrayLike
        Gradient prediction
    """

    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)
        data = np.concatenate((data, np.zeros_like(data)), axis=0)

    data_converted = data.transpose((1, 2, 3, 0))

    if diameter is not None and diameter > 0:
        rescale = model.diam_mean / diameter

    elif rescale is None:
        diameter = model.diam_labels
        rescale = model.diam_mean / diameter

    x = np.asarray(data_converted)

    if normalize:
        # Local normalization, not accessed if global norm is computed
        normalize_default = {
            "lowhigh": None,
            "percentile": None,
            "normalize": normalize,
            "norm3D": False,
            "sharpen_radius": 0,
            "smooth_radius": 0,
            "tile_norm_blocksize": 0,
            "tile_norm_smooth3D": 1,
            "invert": False,
        }
        # Local normalization
        x = transforms.normalize_img(x, **normalize_default)

    elif channel_percentiles is not None:
        # global normalization using global percentiles
        x = percentile_normalization(
            data=x, chn_percentiles=channel_percentiles, channels=np.unique(channels)
        )

    else:
        raise ValueError("Please, check the normalization method.")

    y, style = run_2D_cellpose(
        net=model.net,
        imgs=x,
        img_axis=axis,
        rsz=rescale,
        anisotropy=anisotropy,
        augment=False,
        tile=True,
        tile_overlap=0.1,
        bsize=224,
    )

    return y


def large_scale_cellpose_gradients_per_axis(
    lazy_data: ArrayLike,
    output_gradients_path: PathLike,
    axis: int,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    results_folder: PathLike,
    super_chunksize: Optional[Tuple[int, ...]] = None,
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    global_normalization: Optional[bool] = True,
    model_name: Optional[str] = "cyto",
    cell_diameter: Optional[int] = 15,
    cell_channels: Optional[List[int]] = [0, 0],
    chn_percentiles: Optional[Dict] = None,
):
    """
    Large-scale cellpose prediction of gradients.
    We estimate the gradients using entire 2D planes in
    XY, ZX and ZY directions. Cellpose is in nature a 2D
    network, therefore, there is no degradation in the
    prediction. We save these gradient estimation in
    each axis in a zarr dataset.

    Parameters
    ----------
    lazy_data: ArrayLike
        Loaded lazy dataset.

    output_gradients_path: PathLike
        Path where we want to output the estimated gradients
        in each plane.

    axis: int
        Axis that we are currently using for the estimation.

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize.

    target_size_mb: int
        Parameter used to load a super chunk from the zarr dataset.
        This improves i/o operations and should be bigger than the
        prediction chunksize. Please, verify the amount of available
        shared memory in your system to set this parameter.

    n_workers: int
        Number of workers that will be pulling data from the
        super chunk.

    batch_size: int
        Number of prediction chunksize blocks that will be pulled
        per worker

    results_folder: PathLike
        Path where the results folder for cell segmentation
        is located.

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size. Could be None if target_size_mb is provided.
        Default: None

    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None
        Lazy callback function that will be applied to each of the chunks
        before they are sent to the GPU for prediction. E.g., we might need
        to run deskewing before we run prediction.

    global_normalization: Optional[bool] = True
        If we want to normalize the data for cellpose.

    model_name: Optional[str] = "cyto"
        Model name to be used by cellpose

    cell_diameter: Optional[int] = 15
        Cell diameter for cellpose

    cell_channels: Optional[List[int]]
        List of channels that we are going to use for prediction.
        If background channel is on axis 0 and nuclear in channel 1,
        then cell_channels=[0, 1]. Default: [0, 0]

    chn_percentiles: Optional[Dict]
        Dataset percentiles

    """

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    mode = "a"
    if axis == 0:
        mode = "w"

    logger = utils.create_logger(output_log_path=results_folder, mode=mode)
    logger.info(
        f"{20*'='} Z1 Large-Scale Cellpose Gradient Prediction in Axis {axis} {20*'='}"
    )

    if axis == 0:
        utils.print_system_information(logger)

    logger.info(f"Processing dataset of shape {lazy_data.shape}")

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
    # and preallocate space in GPU
    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    # Overlap between prediction chunks, this overlap happens in every axis
    overlap_prediction_chunksize = tuple([0] * len(prediction_chunksize))

    # Creating zarr data loader
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

    shape = None
    # Creating or reading dataset depending of which axis we're processing
    if axis == 0:
        logger.info(f"Creating zarr gradients in path: {output_gradients_path}")
        output_gradients = zarr.open(
            output_gradients_path,
            "w",
            shape=(
                3,
                3,
            )
            + zarr_dataset.lazy_data.shape[-3:],
            chunks=(
                1,
                3,
            )
            + tuple(prediction_chunksize[-3:]),
            dtype=np.float32,
        )
        shape = zarr_dataset.lazy_data.shape[-3:]

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
    use_GPU = use_gpu()
    logger.info(f"GPU activated: {use_GPU}")
    logger_setup()

    # Getting current GPU device and inizialing cellpose network
    sdevice, gpu = assign_device(use_torch=use_GPU, gpu=use_GPU)
    model = CellposeModel(
        gpu=gpu, model_type=model_name, diam_mean=cell_diameter, device=sdevice
    )

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(
        f"Number of batches: {total_batches} - Samples per iteration: {samples_per_iter}"
    )

    logger.info(
        f"{20*'='} Starting estimation of cellpose combined gradients - Axis {axis} - Channels {cell_channels} {20*'='}"
    )

    local_normalization = False if global_normalization else True
    start_time = time()

    # Processing entire dataset
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

        global_coord_pos = (slice(axis, axis + 1), slice(0, 3)) + global_coord_pos[-3:]

        # Estimating plane gradient
        y = run_cellpose_net(
            data=data,
            model=model,
            axis=axis,
            channels=cell_channels,
            normalize=local_normalization,
            diameter=15,
            anisotropy=1.0,
            channel_percentiles=chn_percentiles,
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

        prediction = np.expand_dims(y, axis=0)

        output_shape = output_gradients[global_coord_pos].shape
        output_prediction = prediction[
            : output_shape[0],
            : output_shape[1],
            : output_shape[2],
            : output_shape[3],
            : output_shape[4],
        ]
        output_gradients[global_coord_pos] = output_prediction

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
            "cellpose_predict_gradients",
        )


def predict_gradients(
    dataset_paths: List[PathLike],
    multiscale: str,
    output_gradients_path: PathLike,
    slices_per_axis: List[int],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    results_folder: PathLike,
    super_chunksize: Optional[Tuple[int, ...]] = None,
    global_normalization: Optional[bool] = True,
    model_name: Optional[str] = "cyto",
    cell_diameter: Optional[int] = 15,
    cell_channels: Optional[List[int]] = [0, 0],
    min_cell_volume: Optional[int] = 95,
    percentile_range: Optional[Tuple[float, float]] = (10, 99),
) -> Tuple[int]:
    """
    Large-scale cellpose prediction of gradients.
    We estimate the gradients using entire 2D planes in
    XY, ZX and ZY directions. Cellpose is in nature a 2D
    network, therefore, there is no degradation in the
    prediction. We save these gradient estimation in
    each axis in a zarr dataset.

    Parameters
    ----------
    dataset_paths: List[PathLike]
        Paths where the datasets in Zarr format are located.
        These could be background channel and nuclei channel in
        as different zarr datasets.
        If the data is in the cloud, please provide the
        path to it. E.g., s3://bucket-name/path/image.zarr

    multiscale: str
        Dataset name insize the zarr dataset. If the zarr
        dataset is not organized in a folder structure,
        please use '.'

    output_gradients_path: PathLike
        Path where we want to output the estimated gradients
        in each plane.

    slices_per_axis: int
        Number of slices that will be pulled each time
        per axis. This should be set up.

    target_size_mb: int
        Parameter used to load a super chunk from the zarr dataset.
        This improves i/o operations and should be bigger than the
        prediction chunksize. Please, verify the amount of available
        shared memory in your system to set this parameter.

    n_workers: int
        Number of workers that will be pulling data from the
        super chunk.

    batch_size: int
        Number of prediction chunksize blocks that will be pulled
        per worker

    results_folder: PathLike
        Path where the results folder for cell segmentation
        is located.

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size. Could be None if target_size_mb is provided.
        Default: None

    global_normalization: Optional[bool] = True
        If we want to normalize the data for cellpose.

    model_name: Optional[str] = "cyto"
        Model name to be used by cellpose

    cell_diameter: Optional[int] = 15
        Cell diameter for cellpose

    cell_channels: Optional[List[int]]
        List of channels that we are going to use for prediction.
        If background channel is on axis 0 and nuclear in channel 1,
        then cell_channels=[0, 1]. Default: [0, 0]

    Returns
    -------
    Tuple[int]
        Tuple with the shape of the original dataset
    """
    len_datasets = len(dataset_paths)
    lazy_data = None

    if not len_datasets:
        raise ValueError("Empty list of datasets!")

    elif len_datasets:
        lazy_data = concatenate_lazy_data(
            dataset_paths=dataset_paths,
            multiscale=multiscale,
            concat_axis=-4,  # Concatenation axis
        )
        print("Combined background and nuclear channel: ", lazy_data, lazy_data.dtype)

    combined_percentiles = None
    if global_normalization:
        print(f"Computing global percentiles...")
        combined_percentiles, chunked_percentiles = compute_percentiles(
            lazy_data=lazy_data,
            target_size_mb=target_size_mb,
            percentile_range=percentile_range,
            min_cell_volume=min_cell_volume,
            n_workers=16,
            threads_per_worker=1,
            combine_method="median",
        )
        print("Estimated global percentiles: ", combined_percentiles)
        np.save(f"{results_folder}/combined_percentiles.npy", combined_percentiles)
        np.save(f"{results_folder}/chunked_percentiles.npy", chunked_percentiles)

    # Reading image shape
    image_shape = lazy_data.shape
    factor = 6

    # axes_names = ["XY", "ZX", "ZY"]

    # Processing each plane at a time. This could be faster if you have more
    # GPUs, we are currently running this on a single GPU machine.
    for axis in range(0, 3):

        slice_per_axis = slices_per_axis[axis]
        prediction_chunksize = None
        super_chunksize = None

        # Setting prediction chunksize to entire planes using the number of slices per axis
        if axis == 0:
            prediction_chunksize = (slice_per_axis, image_shape[-2], image_shape[-1])
            super_chunksize = (
                slice_per_axis * factor,
                image_shape[-2],
                image_shape[-1],
            )

        elif axis == 1:
            prediction_chunksize = (image_shape[-3], slice_per_axis, image_shape[-1])
            super_chunksize = (
                image_shape[-3],
                slice_per_axis * factor,
                image_shape[-1],
            )

        elif axis == 2:
            prediction_chunksize = (image_shape[-3], image_shape[-2], slice_per_axis)
            super_chunksize = (
                image_shape[-3],
                image_shape[-2],
                slice_per_axis * factor,
            )

        # Adding the channels to the prediction chunksize
        if len_datasets > 1:
            prediction_chunksize = (len_datasets,) + prediction_chunksize
            super_chunksize = (len_datasets,) + super_chunksize

        large_scale_cellpose_gradients_per_axis(
            lazy_data=lazy_data,
            output_gradients_path=output_gradients_path,
            axis=axis,
            prediction_chunksize=prediction_chunksize,
            target_size_mb=target_size_mb,
            n_workers=n_workers,
            batch_size=batch_size,
            super_chunksize=super_chunksize,
            global_normalization=global_normalization,
            model_name=model_name,
            cell_diameter=cell_diameter,
            results_folder=results_folder,
            cell_channels=cell_channels,
            chn_percentiles=combined_percentiles,
        )

    return image_shape[-3:]


def main():
    """
    Main function
    """
    BUCKET_NAME = "aind-open-data"
    IMAGE_PATH = "HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49"
    TILE_NAME = "channel_405.zarr"
    # dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    dataset_path = f"/data/{IMAGE_PATH}/{TILE_NAME}"

    # Data loader params
    super_chunksize = None
    target_size_mb = 3072  # None
    n_workers = 16
    batch_size = 1

    # Cellpose params
    model_name = "cyto"
    global_normalization = True  # TODO Normalize image in the entire dataset
    cell_diameter = 15

    slices_per_axis = [40, 80, 80]

    # output gradients
    output_gradients_path = "../results/gradients.zarr"

    # Large-scale prediction of gradients
    predict_gradients(
        dataset_paths=[dataset_path],
        multiscale="2",
        output_gradients_path=output_gradients_path,
        slices_per_axis=slices_per_axis,
        target_size_mb=target_size_mb,
        n_workers=n_workers,
        batch_size=batch_size,
        super_chunksize=super_chunksize,
        global_normalization=global_normalization,
        model_name=model_name,
        cell_diameter=cell_diameter,
        cell_channels=[0, 0],  # RN28s and Nuclei
    )


if __name__ == "__main__":
    main()
