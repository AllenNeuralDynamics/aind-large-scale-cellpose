"""
Code to upsample a segmentatation mask
"""

import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numcodecs
import numpy as np
import s3fs
import xarray_multiscale
import zarr
from dask.distributed import Client, LocalCluster
from numcodecs import Blosc
from zarr import Group, open_group

from . import utils
from .omezarr_metadata import _get_pyramid_metadata, write_ome_ngff_metadata
from .zarr_writer import BlockedArrayWriter


def compute_pyramid(
    data: dask.array.core.Array,
    n_lvls: int,
    scale_axis: Tuple[int],
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
) -> List[dask.array.core.Array]:
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mode_countless,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid]


def write_multiscales(
    path_to_data: Union[str, Path],
    voxel_size: List[float],
    chunk_size: List[int] = [128, 128, 128],
    scale_factors_per_level: List[List[int]] = None,
    target_size_mb: int = 2048,
    n_lvls: int = None,
    root_group: Group = None,
):
    """
    Writes a multi-scale pyramid from an existing Zarr dataset.

    Parameters
    ----------
    path_to_data : Union[str, Path]
        Path to the base Zarr dataset (e.g., '0' level should be present).
    chunk_size : List[int], optional
        Chunk size to use for writing each pyramid level. Default is [128, 128, 128].
    scale_factors_per_level : List[List[int]]
        Downsampling scale factor for each pyramid level transition, in ZYX
        order. E.g. [[1,2,2], [2,2,2]] means level 0→1 uses [1,2,2] and
        level 1→2 uses [2,2,2]. Length determines the number of levels written
        unless overridden by ``n_lvls``.
    target_size_mb : int, optional
        Target block size in MB for optimized writing. Default is 2048 MB.
    n_lvls : int, optional
        Number of pyramid levels to generate (excluding base). Defaults to
        ``len(scale_factors_per_level)``.
    root_group : Group, optional
        Zarr group to write the pyramid to. If None, a new group will be created at `path_to_data`.
    """
    if scale_factors_per_level is None:
        raise ValueError("scale_factors_per_level must be provided.")
    if n_lvls is None:
        n_lvls = len(scale_factors_per_level)
    path_to_data = Path(path_to_data)
    if not path_to_data.exists():
        raise FileNotFoundError(f"Path {path_to_data} does not exist!")

    # Load the base scale (level 0)
    base_scale = da.from_zarr(path_to_data / "0")

    if root_group is None:
        # Assume top-level group creation if not provided
        root_group = open_group(path_to_data.parent, mode="a")

    if path_to_data.name in root_group:
        new_channel_group = root_group[path_to_data.name]
        print(f"Group '{path_to_data.name}' already exists. Reusing it.")
    else:
        raise ValueError("There must be a group created!")

    # Compute block shape used for optimized writing
    block_shape = list(
        BlockedArrayWriter.get_block_shape(
            arr=base_scale,
            target_size_mb=target_size_mb,
            chunks=chunk_size,
        )
    )

    # Pad block shape if fewer than 5D
    extra_axes = (1,) * (5 - len(block_shape))

    block_shape = extra_axes + tuple(block_shape)

    extra_axes_chunks = (1,) * (5 - len(chunk_size))
    chunk_size = extra_axes_chunks + tuple(chunk_size)

    multiscale_zarr_json = write_ome_ngff_metadata(
        arr_shape=base_scale.shape,
        chunk_size=chunk_size,
        image_name="cell_segmentation",
        n_lvls=n_lvls,
        scale_factors=scale_factors_per_level,
        voxel_size=voxel_size,
        origin=[0, 0, 0],
        metadata=_get_pyramid_metadata(),
    )

    with open(f"{path_to_data}/.zattrs", "w") as f:
        json.dump(multiscale_zarr_json, f)

    # Compression settings
    compressor = Blosc(cname="zstd", clevel=3, shuffle=1, blocksize=0)

    current_scale = base_scale

    for level in range(n_lvls):
        # Use the scale factor for this specific level transition; fall back to
        # the last entry if n_lvls exceeds the length of the list.
        level_factor = scale_factors_per_level[min(level, len(scale_factors_per_level) - 1)]

        # Add missing dimensions if needed
        scale_factors_padded = ([1] * (len(current_scale.shape) - len(level_factor))) + level_factor

        # Compute one level of pyramid
        pyramid = compute_pyramid(
            data=current_scale,
            scale_axis=scale_factors_padded,
            chunks=chunk_size,
            n_lvls=2,  # Generate next level only
        )

        # Select the downsampled array (next level)
        current_scale = pyramid[-1]

        print(
            f"[level {level + 1}] Writing pyramid level with shape {current_scale.shape} - Block shape: {block_shape}"
        )

        # Create Zarr dataset for the level
        pyramid_group = new_channel_group.create_dataset(
            name=str(level + 1),
            shape=current_scale.shape,
            chunks=chunk_size,
            dtype=current_scale.dtype,
            compressor=compressor,
            dimension_separator="/",
            overwrite=True,
        )

        # Store data in blocks
        BlockedArrayWriter.store(current_scale, pyramid_group, block_shape)


def initialize_output_volume(
    output_params: Dict,
    output_volume_size: Tuple[int, int, int],
) -> zarr.core.Array:
    """
    Initializes the zarr directory where the
    volume will be upsampled.

    Inputs
    ------
    output_params: Dict
        Parameters to create the zarr storage.
    output_volume_size: Tuple[int]
        Output volume size for the zarr file.

    Returns
    -------
    Zarr thread-safe datastore initialized on OutputParameters.
    """

    # Local execution
    out_group = zarr.open_group(output_params["path"], mode="w")

    # Cloud execuion
    if output_params["path"].startswith("s3"):
        s3 = s3fs.S3FileSystem(
            config_kwargs={
                "max_pool_connections": 50,
                "s3": {
                    "multipart_threshold": 64
                    * 1024
                    * 1024,  # 64 MB, avoid multipart upload for small chunks
                    "max_concurrent_requests": 20,  # Increased from 10 -> 20.
                },
                "retries": {
                    "total_max_attempts": 100,
                    "mode": "adaptive",
                },
            }
        )
        store = s3fs.S3Map(root=output_params["path"], s3=s3)
        out_group = zarr.open(store=store, mode="a")

    path = "0"
    chunksize = output_params["chunksize"]
    datatype = output_params["dtype"]
    dimension_separator = output_params["dimension_separator"]
    compressor = output_params["compressor"]
    print("Using compressor: ", compressor)
    output_volume = out_group.create_dataset(
        path,
        shape=(
            1,
            1,
            output_volume_size[0],
            output_volume_size[1],
            output_volume_size[2],
        ),
        chunks=chunksize,
        dtype=datatype,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=True,
        fill_value=0,
    )

    return output_volume


def upscale_zarr_with_padding(
    input_zarr,
    output_params: Dict,
    upscale_factors_zyx: Tuple[int] = (1, 4, 4),
    new_shape: Optional[Tuple] = None,
    n_workers: Optional[int] = 16,
):
    """
    Upscale a Zarr volume by specified factors in the spatial dimensions (z, y, x)
    and save to a new Zarr file. Assumes input is in tczyx format with t and c = 1.
    Adds zero padding if new_shape is provided and differs from calculated shape.

    Parameters:
    input_zarr: dask.array.Array
        Lazy mask
    output_params: Dict
        Dictionary with the parameters for the output Zarr file.
    upscale_factors_zyx: Tuple[int]
        Tuple of upscale factors for z, y, and x dimensions. Default: (1, 4, 4)
    new_shape: Optional[Tuple]
        If provided, the output will be padded to this shape. Default: None
    n_workers: Optional[int]
        Optional number of workers for the dask cluster

    """
    t, c = 1, 1
    if len(input_zarr.shape) == 5:
        _, _, z, y, x = input_zarr.shape
    else:
        z, y, x = input_zarr.shape

    # Calculate the shape of the upscaled volume
    calculated_new_shape = (
        t,
        c,
        z * upscale_factors_zyx[0],
        y * upscale_factors_zyx[1],
        x * upscale_factors_zyx[2],
    )

    if new_shape is not None:
        if len(new_shape) != 5:
            new_shape = (t, c, new_shape[0], new_shape[1], new_shape[2])
        padding = tuple(max(0, new - calc) for new, calc in zip(new_shape, calculated_new_shape))
    else:
        new_shape = calculated_new_shape
        padding = (0, 0, 0, 0, 0)

    chunk_size = output_params["chunksize"]  # (1, 1, 128, 128, 128)

    print("Getting max value")
    max_value = input_zarr.max()
    print("Maximum value:", max_value)

    print(
        f"Upscaling from size {input_zarr.shape} by {upscale_factors_zyx} to new shape {new_shape} with {output_params['chunksize']} chunk size and dtype: {output_params['dtype']} as determined by maximum value {max_value}"
    )
    print(f"Padding: {padding}")

    client = Client(LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True))

    # Initialize output volume with the new shape
    output_zarr = initialize_output_volume(output_params, new_shape[-3:])

    # Calculate the total number of chunks to process
    total_chunks = (np.ceil(z / 128) * np.ceil(y / 128) * np.ceil(x / 128)).astype(int)
    current_chunk = 1

    # Process and upscale each chunk
    for z_idx in range(0, z, 128):
        for y_idx in range(0, y, 128):
            for x_idx in range(0, x, 128):
                current_chunk += 1
                # Extract the current chunk
                if len(input_zarr.shape) == 5:  # tczyx
                    chunk = input_zarr[
                        0,
                        0,
                        z_idx : z_idx + 128,
                        y_idx : y_idx + 128,
                        x_idx : x_idx + 128,
                    ]
                elif len(input_zarr.shape) == 3:  # zyx
                    chunk = input_zarr[
                        z_idx : z_idx + 128,
                        y_idx : y_idx + 128,
                        x_idx : x_idx + 128,
                    ]
                else:
                    print(
                        "len(input_zarr.shape) not compatible: ",
                        len(input_zarr.shape),
                        "exiting",
                    )
                    exit()

                # Upscale the chunk by duplicating each value to fill a block
                upscaled_chunk = np.repeat(
                    np.repeat(
                        np.repeat(chunk, upscale_factors_zyx[0], axis=0),
                        upscale_factors_zyx[1],
                        axis=1,
                    ),
                    upscale_factors_zyx[2],
                    axis=2,
                )

                # Calculate the indices for placing the upscaled chunk in the output
                z_new, y_new, x_new = (
                    z_idx * upscale_factors_zyx[0],
                    y_idx * upscale_factors_zyx[1],
                    x_idx * upscale_factors_zyx[2],
                )
                print(
                    f"Processing chunk {current_chunk}/{total_chunks} at z: {z_new}, y: {y_new}, x: {x_new}"
                    f" - Upscaled chunk shape: {upscaled_chunk.shape} - {new_shape} new shape"
                )

                # Add the upscaled chunk to the output, considering padding
                output_zarr[
                    0,
                    0,
                    z_new : min(
                        z_new + upscaled_chunk.shape[0],
                        new_shape[2] - padding[2],
                    ),
                    y_new : min(
                        y_new + upscaled_chunk.shape[1],
                        new_shape[3] - padding[3],
                    ),
                    x_new : min(
                        x_new + upscaled_chunk.shape[2],
                        new_shape[4] - padding[4],
                    ),
                ] = upscaled_chunk[
                    : min(
                        upscaled_chunk.shape[0],
                        new_shape[2] - padding[2] - z_new,
                    ),
                    : min(
                        upscaled_chunk.shape[1],
                        new_shape[3] - padding[3] - y_new,
                    ),
                    : min(
                        upscaled_chunk.shape[2],
                        new_shape[4] - padding[4] - x_new,
                    ),
                ]

    print("Upscaling completed.")


def upscale_array_3d(data_3d: np.ndarray, upscale_factors_zyx: Tuple[int]) -> np.ndarray:
    """
    Upscale a 3D array using nearest neighbor interpolation (repeat).
    """
    upscaled = data_3d

    factors = tuple(int(f) for f in upscale_factors_zyx)

    if factors[0] > 1:
        upscaled = np.repeat(upscaled, factors[0], axis=0)
    if factors[1] > 1:
        upscaled = np.repeat(upscaled, factors[1], axis=1)
    if factors[2] > 1:
        upscaled = np.repeat(upscaled, factors[2], axis=2)

    return upscaled


def _upscale_chunk(
    chunk_idx: int,
    data_3d: np.ndarray,
    chunk_size_z: int,
    z_original: int,
    upscale_factors_zyx: Tuple,
    zarr_array: zarr.core.Array,
    output_dtype: np.dtype,
) -> Tuple[int, str]:
    """Process and write a single Z chunk. Safe to call concurrently as long
    as each chunk_idx maps to a non-overlapping Z range in zarr_array."""
    z_start = chunk_idx * chunk_size_z
    z_end = min((chunk_idx + 1) * chunk_size_z, z_original)

    chunk_data = np.asarray(data_3d[z_start:z_end, :, :])
    upscaled_chunk = upscale_array_3d(chunk_data, upscale_factors_zyx)

    out_z_start = int(z_start * upscale_factors_zyx[0])
    out_z_end = min(int(z_end * upscale_factors_zyx[0]), zarr_array.shape[2])

    if out_z_start >= zarr_array.shape[2]:
        return chunk_idx, "skipped"

    actual_z_size = out_z_end - out_z_start
    upscaled_chunk = upscaled_chunk[:actual_z_size, :, :]

    if upscaled_chunk.dtype != output_dtype:
        upscaled_chunk = upscaled_chunk.astype(output_dtype)

    coords = (
        slice(None),
        slice(None),
        slice(out_z_start, out_z_end),
        slice(0, upscaled_chunk.shape[1]),
        slice(0, upscaled_chunk.shape[2]),
    )
    zarr_array[coords] = upscaled_chunk[np.newaxis, np.newaxis, :, :, :]
    return chunk_idx, "done"


def upscale_zarr_with_padding_chunked(
    input_data: np.ndarray,
    output_params: Dict,
    upscale_factors_zyx: Tuple[int] = (1, 4, 4),
    new_shape: Optional[Tuple] = None,
    chunk_size_z: int = 32,
    n_workers: int = 4,
):
    """
    Memory-efficient upscaling using chunked processing.

    Parameters:
    -----------
    chunk_size_z: int
        Number of Z slices to process at once. Reduce if still running out of memory.
    n_workers: int
        Number of parallel threads for chunk processing. Default: 4.
    """

    # Handle input dimensions
    if len(input_data.shape) == 5:
        t, c, z, y, x = input_data.shape
        if t != 1 or c != 1:
            raise ValueError(f"Expected t=1, c=1 for 5D input, got t={t}, c={c}")
        data_3d = input_data[0, 0]
    elif len(input_data.shape) == 3:
        z, y, x = input_data.shape
        data_3d = input_data
        t, c = 1, 1
    else:
        raise ValueError(
            f"Input must be 3D (z,y,x) or 5D (t,c,z,y,x), got shape {input_data.shape}"
        )

    print(f"Input shape: {input_data.shape}")
    print(f"Processing 3D volume: {data_3d.shape}")

    upscale_factors_zyx = tuple(int(f) for f in upscale_factors_zyx)

    # Calculate output dimensions
    upscaled_shape_3d = (
        z * upscale_factors_zyx[0],
        y * upscale_factors_zyx[1],
        x * upscale_factors_zyx[2],
    )
    print(f"Upscaled shape: ", upscaled_shape_3d)

    # Handle padding
    if new_shape is not None:
        if len(new_shape) == 3:
            target_shape_3d = new_shape
        else:
            raise ValueError("new_shape must be 3D (z,y,x)")

        padding_3d = tuple(
            max(0, target - upscaled)
            for target, upscaled in zip(target_shape_3d, upscaled_shape_3d)
        )

        if any(p > 0 for p in padding_3d):
            print(f"Padding will be added: {padding_3d}")
    else:
        target_shape_3d = upscaled_shape_3d
        padding_3d = (0, 0, 0)

    # Get output dtype
    max_value = data_3d.max()
    if max_value <= np.iinfo(np.uint8).max:
        output_dtype = np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        output_dtype = np.uint16
    else:
        output_dtype = np.uint32

    if "dtype" in output_params:
        output_dtype = np.dtype(output_params["dtype"])

    print(f"Target shape: {target_shape_3d}")
    print(f"Output dtype: {output_dtype}")
    print(f"Processing in chunks of {chunk_size_z} Z slices")

    # Create output zarr array first
    output_path = output_params.get("path", "output.zarr")
    compression = output_params.get("compression", "zstd")
    compression_opts = output_params.get("compression_opts", 3)
    default_chunks = (
        1,
        1,
        min(128, target_shape_3d[0]),
        min(128, target_shape_3d[1]),
        min(128, target_shape_3d[2]),
    )
    chunks = output_params.get("chunksize", default_chunks)

    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)

    # Create zarr array with final shape
    final_shape = (1, 1) + target_shape_3d
    print("Final shape: ", final_shape)
    zarr_array = root.create_dataset(
        "0",
        shape=final_shape,
        dtype=output_dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        overwrite=True,
        fill_value=0,
        dimension_separator="/",
    )

    # Process chunks in parallel — each chunk writes to a non-overlapping
    # Z range so concurrent zarr writes are safe.
    z_original = data_3d.shape[0]
    num_chunks = (z_original + chunk_size_z - 1) // chunk_size_z
    print(f"Processing {num_chunks} chunks with {n_workers} workers")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _upscale_chunk,
                chunk_idx,
                data_3d,
                chunk_size_z,
                z_original,
                upscale_factors_zyx,
                zarr_array,
                output_dtype,
            ): chunk_idx
            for chunk_idx in range(num_chunks)
        }
        for future in as_completed(futures):
            chunk_idx, status = future.result()
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: {status}")

    gc.collect()
    print("Chunked upscaling completed successfully!")
    return target_shape_3d, output_dtype


def upscale_mask(
    dataset_path: str,
    mask_data: np.ndarray,
    output_folder: str,
    upscale_factors_zyx: Optional[tuple] = None,
    filename: Optional[str] = "segmentation_mask.zarr",
    dest_multiscale: Optional[str] = "0",
    source_multiscale: Optional[str] = None,
    n_workers: int = 4,
):
    """
    Upscales a segmentation mask

    Parameters
    ----------
    dataset_path: str
        Path where the dataset that was used
        for segmentation is located.
    mask_data: np.ndarray
        The segmentation mask to upscale.
    output_folder: str
        Path where the upsampled segmentation mask will
        be stored.
    upscale_factors_zyx: Optional[tuple]
        Per-axis (Z, Y, X) upscale factors. If None and
        source_multiscale is provided, factors are derived
        automatically from the OME-Zarr coordinate
        transformations metadata.
    filename: str
        Filename for the upsampled segmentation mask
    dest_multiscale: Optional[str]
        Destination multiscale level (target resolution).
        Default: "0"
    source_multiscale: Optional[str]
        Source multiscale level that the mask was segmented at.
        When provided, per-axis upscale factors are computed
        as source_scale / dest_scale for each spatial axis,
        correctly handling anisotropic pyramids where Z and
        XY may have different downsampling factors per level.
    """
    output_folder = Path(output_folder)

    if not output_folder.exists():
        raise FileNotFoundError(f"The output folder {output_folder} does not exist!")

    raw_metadata = utils.load_json(data_path=dataset_path, keyname=".zattrs")
    image_lazy_data = da.from_zarr(f"{dataset_path}/0")

    image_compressor = raw_metadata.get(".zarray", {}).get("compressor", None)
    if image_compressor is None:
        print("Image metadata does not contain a valid compressor. Please check the dataset.")

    multiscales = raw_metadata.get("multiscales", [])

    if not len(multiscales):
        raise ValueError(f"We need to have a multiscale pyramid dataset. Metadata: {raw_metadata}")

    pyramid_scales = multiscales[0].get("datasets", [])

    if not len(pyramid_scales) > 1:
        raise ValueError(f"We need to have multiple scales. Metadata: {raw_metadata}")

    dest_metadata = utils.parse_zarr_metadata(metadata=raw_metadata, multiscale=dest_multiscale)

    # Getting list with Z Y X order of the resolution
    resolution_zyx = (
        dest_metadata["axes"]["z"]["scale"],
        dest_metadata["axes"]["y"]["scale"],
        dest_metadata["axes"]["x"]["scale"],
    )

    # Derive per-axis upscale factors from OME-Zarr metadata when
    # source_multiscale is given. This correctly handles anisotropic
    # pyramids where Z and XY downsampling factors differ per level.
    if source_multiscale is not None:
        source_metadata = utils.parse_zarr_metadata(
            metadata=raw_metadata, multiscale=source_multiscale
        )
        upscale_factors_zyx = tuple(
            source_metadata["axes"][ax]["scale"] / dest_metadata["axes"][ax]["scale"]
            for ax in ["z", "y", "x"]
        )
        print(
            "Derived upscale_factors_zyx from metadata "
            f"(source level {source_multiscale} -> dest level {dest_multiscale}): "
            f"{upscale_factors_zyx}"
        )
    elif upscale_factors_zyx is None:
        raise ValueError("Either upscale_factors_zyx or source_multiscale must be provided.")

    # Compute per-level scale factors for every consecutive transition in the
    # original pyramid. This mirrors the exact anisotropic structure of the
    # source data — e.g. Z may use a different factor than Y/X, and the
    # factor can differ between levels (level 0→1 vs level 1→2, etc.).
    per_level_scale_factors = []
    for i in range(len(pyramid_scales) - 1):
        src_path = pyramid_scales[i]["path"]
        dst_path = pyramid_scales[i + 1]["path"]
        src_meta = utils.parse_zarr_metadata(metadata=raw_metadata, multiscale=src_path)
        dst_meta = utils.parse_zarr_metadata(metadata=raw_metadata, multiscale=dst_path)
        factor = [
            round(dst_meta["axes"][ax]["scale"] / src_meta["axes"][ax]["scale"])
            for ax in ["z", "y", "x"]
        ]
        per_level_scale_factors.append(factor)

    print(f"Per-level scale factors from metadata: {per_level_scale_factors}")

    # Add this check and conversion:
    if isinstance(image_compressor, dict):
        if image_compressor["id"] == "blosc":
            image_compressor = numcodecs.Blosc(
                cname=image_compressor.get("cname", "zstd"),
                clevel=image_compressor.get("clevel", 1),
                shuffle=image_compressor.get("shuffle", 1),
                blocksize=image_compressor.get("blocksize", 0),
            )
        else:
            # Default fallback if compressor type is unknown
            image_compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    elif image_compressor is None:
        image_compressor = numcodecs.Blosc(cname="zstd", clevel=3)

    output_filepath = output_folder.joinpath(filename).as_posix()
    output_params = {
        "chunksize": [1, 1, 128, 128, 128],
        "resolution_zyx": resolution_zyx,
        "dtype": np.uint8,
        "path": output_filepath,
        "compressor": image_compressor,
        "dimension_separator": "/",
    }

    upscale_zarr_with_padding_chunked(
        input_data=mask_data,
        output_params=output_params,
        upscale_factors_zyx=upscale_factors_zyx,
        new_shape=image_lazy_data.shape[-3:],
        chunk_size_z=32,
        n_workers=n_workers,
    )

    return resolution_zyx, len(pyramid_scales), per_level_scale_factors
