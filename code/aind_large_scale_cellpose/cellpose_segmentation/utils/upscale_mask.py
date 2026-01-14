"""
Code to upsample a segmentatation mask
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import dask
import numcodecs
import numpy as np
import s3fs
import zarr
from aind_large_scale_cellpose.cellpose_segmentation.utils import utils
from aind_large_scale_prediction.io import ImageReaderFactory, extract_data
from dask import delayed
from dask.distributed import Client, LocalCluster, performance_report


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
    max_value = input_zarr.max().compute()
    print("Maximum value:", max_value)

    # Set output_dtype based on the maximum value
    if max_value <= np.iinfo(np.uint16).max:
        output_dtype = "uint16"
        output_params["dtype"] = "uint16"
    else:
        output_dtype = "uint32"
        output_params["dtype"] = "uint32"

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
                        0, 0, z_idx : z_idx + 128, y_idx : y_idx + 128, x_idx : x_idx + 128
                    ]
                elif len(input_zarr.shape) == 3:  # zyx
                    chunk = input_zarr[
                        z_idx : z_idx + 128, y_idx : y_idx + 128, x_idx : x_idx + 128
                    ]
                else:
                    print(
                        "len(input_zarr.shape) not compatible: ", len(input_zarr.shape), "exiting"
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

                # Add the upscaled chunk to the output, considering padding
                output_zarr[
                    0,
                    0,
                    z_new : min(z_new + upscaled_chunk.shape[0], new_shape[2] - padding[2]),
                    y_new : min(y_new + upscaled_chunk.shape[1], new_shape[3] - padding[3]),
                    x_new : min(x_new + upscaled_chunk.shape[2], new_shape[4] - padding[4]),
                ] = upscaled_chunk[
                    : min(upscaled_chunk.shape[0], new_shape[2] - padding[2] - z_new),
                    : min(upscaled_chunk.shape[1], new_shape[3] - padding[3] - y_new),
                    : min(upscaled_chunk.shape[2], new_shape[4] - padding[4] - x_new),
                ].compute()

    print("Upscaling completed.")


def generate_multiscale_pyramid(
    zarr_path: Union[str, Path],
    max_level: int = 5,
    downsample_factor: int = 2,
    n_workers: Optional[int] = 16,
    chunk_size: Tuple[int, int, int, int, int] = (1, 1, 128, 128, 128),
) -> None:
    """
    Generate a multiscale pyramid for a segmentation mask stored in zarr format.
    
    Creates pyramid levels 1 through max_level by iteratively downsampling the
    previous level. Uses mode-based downsampling to preserve segmentation labels.
    Each level is stored in the same zarr group as "{level}" where level 0 is
    the full resolution.
    
    This function follows OME-NGFF conventions for multiscale image storage
    and is optimized for large-scale microscopy data processing.
    
    Parameters
    ----------
    zarr_path : Union[str, Path]
        Path to the zarr group containing level "0" (full resolution)
    max_level : int, default=5
        Maximum pyramid level to generate (e.g., 5 creates levels 1-5)
    downsample_factor : int, default=2
        Factor by which to downsample each spatial dimension per level
        (typically 2, meaning each level is 1/8 the volume of the previous)
    n_workers : Optional[int], default=16
        Number of dask workers for parallel processing
    chunk_size : Tuple[int, int, int, int, int], default=(1, 1, 128, 128, 128)
        Chunk size for zarr arrays in (t, c, z, y, x) format
        
    Returns
    -------
    None
        Writes pyramid levels directly to the zarr group
        
    Raises
    ------
    FileNotFoundError
        If the zarr path does not exist or level "0" is not found
    ValueError
        If the input data has incompatible dimensions
        
    Notes
    -----
    - Level 0 must already exist in the zarr group (created by upscale_mask)
    - Each pyramid level is created sequentially from the previous level
    - Uses mode-based downsampling to preserve label identity
    - Processing is done in chunks for memory efficiency
    - Compressor and dtype are inherited from level 0
    
    Examples
    --------
    After creating a full resolution mask:
    
    >>> generate_multiscale_pyramid(
    ...     zarr_path="/path/to/segmentation_mask.zarr",
    ...     max_level=5,
    ...     n_workers=16
    ... )
    
    This creates levels 1-5, where each level is downsampled by 2x in z, y, x.
    Level 1 shape: (t, c, z/2, y/2, x/2)
    Level 5 shape: (t, c, z/32, y/32, x/32)
    """
    zarr_path = Path(zarr_path)
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")
    
    # Open the zarr group
    if zarr_path.as_posix().startswith("s3"):
        s3 = s3fs.S3FileSystem(
            config_kwargs={
                "max_pool_connections": 50,
                "s3": {
                    "multipart_threshold": 64 * 1024 * 1024,
                    "max_concurrent_requests": 20,
                },
                "retries": {
                    "total_max_attempts": 100,
                    "mode": "adaptive",
                },
            }
        )
        store = s3fs.S3Map(root=zarr_path.as_posix(), s3=s3)
        root = zarr.open(store=store, mode="a")
    else:
        root = zarr.open_group(zarr_path.as_posix(), mode="a")
    
    # Verify level 0 exists
    if "0" not in root:
        raise ValueError(f"Level '0' not found in zarr group at {zarr_path}")
    
    # Get level 0 metadata
    level0 = root["0"]
    compressor = level0.compressor
    dtype = level0.dtype
    dimension_separator = level0._dimension_separator
    
    print(f"Starting pyramid generation for {zarr_path}")
    print(f"Level 0 shape: {level0.shape}, dtype: {dtype}")
    print(f"Generating levels 1 through {max_level}")
    
    # Initialize dask cluster
    client = Client(LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True))
    
    # Generate each pyramid level
    for level in range(1, max_level + 1):
        print(f"\n{'='*60}")
        print(f"Generating level {level}")
        print(f"{'='*60}")
        
        # Get the previous level
        prev_level = root[str(level - 1)]
        prev_shape = prev_level.shape  # (t, c, z, y, x)
        
        # Calculate new shape (downsample spatial dimensions only)
        new_shape = (
            prev_shape[0],  # t unchanged
            prev_shape[1],  # c unchanged
            prev_shape[2] // downsample_factor,  # z downsampled
            prev_shape[3] // downsample_factor,  # y downsampled
            prev_shape[4] // downsample_factor,  # x downsampled
        )
        
        print(f"Previous level {level-1} shape: {prev_shape}")
        print(f"New level {level} shape: {new_shape}")
        
        # Create the new pyramid level
        current_level = root.create_dataset(
            str(level),
            shape=new_shape,
            chunks=chunk_size,
            dtype=dtype,
            compressor=compressor,
            dimension_separator=dimension_separator,
            overwrite=True,
            fill_value=0,
        )
        
        # Process the data in chunks to downsample
        # We'll read larger chunks from the previous level and write to current level
        read_chunk_z, read_chunk_y, read_chunk_x = 256, 256, 256
        
        total_z_chunks = int(np.ceil(prev_shape[2] / read_chunk_z))
        total_y_chunks = int(np.ceil(prev_shape[3] / read_chunk_y))
        total_x_chunks = int(np.ceil(prev_shape[4] / read_chunk_x))
        total_chunks = total_z_chunks * total_y_chunks * total_x_chunks
        
        print(f"Processing {total_chunks} chunks...")
        
        chunk_counter = 0
        for z_idx in range(0, prev_shape[2], read_chunk_z):
            for y_idx in range(0, prev_shape[3], read_chunk_y):
                for x_idx in range(0, prev_shape[4], read_chunk_x):
                    chunk_counter += 1
                    
                    if chunk_counter % 10 == 0:
                        print(f"  Processing chunk {chunk_counter}/{total_chunks}")
                    
                    # Read chunk from previous level
                    z_end = min(z_idx + read_chunk_z, prev_shape[2])
                    y_end = min(y_idx + read_chunk_y, prev_shape[3])
                    x_end = min(x_idx + read_chunk_x, prev_shape[4])
                    
                    chunk = prev_level[0, 0, z_idx:z_end, y_idx:y_end, x_idx:x_end]
                    
                    # Downsample the chunk
                    downsampled_chunk = downsample_segmentation_block(
                        chunk, factor=downsample_factor
                    )
                    
                    # Calculate output indices
                    z_out = z_idx // downsample_factor
                    y_out = y_idx // downsample_factor
                    x_out = x_idx // downsample_factor
                    
                    # Write to current level
                    current_level[
                        0,
                        0,
                        z_out : z_out + downsampled_chunk.shape[0],
                        y_out : y_out + downsampled_chunk.shape[1],
                        x_out : x_out + downsampled_chunk.shape[2],
                    ] = downsampled_chunk
        
        print(f"Level {level} completed")
    
    client.close()
    print(f"\n{'='*60}")
    print(f"Pyramid generation completed successfully!")
    print(f"Generated levels 0-{max_level} in {zarr_path}")
    print(f"{'='*60}\n")

def upscale_mask(
    dataset_path: str,
    segmentation_mask_path: str,
    output_folder: str,
    filename: Optional[str] = "segmentation_mask.zarr",
    dest_multiscale: Optional[str] = "0",
    n_workers: Optional[int] = 16,
):
    """
    Upscales a segmentation mask

    Parameters
    ----------
    dataset_path: str
        Path where the dataset that was used
        for segmentation is located.
    segmentation_mask_path: str
        Path where the segmentation mask is located
    output_folder: str
        Path where the upsampled segmentation mask will
        be stored.
    filename: str
        Filename for the upsampled segmentation mask
    dest_multiscale: Optional[str]
        Destination multiscale. This is useful to pull
        metadata. Default: "0"
    n_workers: Optional[int]
        Optional number of workers for the dask cluster. Default: 16
    """
    output_folder = Path(output_folder)

    if not output_folder.exists():
        raise FileNotFoundError(f"The output folder {output_folder} does not exist!")

    # Reading image metadata
    img_reader = ImageReaderFactory().create(
        data_path=dataset_path,
        parse_path=False,
        multiscale=dest_multiscale,
    )
    image_lazy_data = extract_data(img_reader.as_dask_array())
    image_metadata = img_reader.metadata()
    image_compressor = image_metadata[".zarray"]["compressor"]
    image_metadata = utils.parse_zarr_metadata(metadata=image_metadata, multiscale=dest_multiscale)

    image_shape = image_lazy_data.shape

    # Getting list with Z Y X order of the resolution
    resolution_zyx = (
        image_metadata["axes"]["z"]["scale"],
        image_metadata["axes"]["y"]["scale"],
        image_metadata["axes"]["x"]["scale"],
    )

    print(
        "Image metadata: ",
        image_metadata,
        " - Resolution: ",
        resolution_zyx,
        " - Image compressor: ",
        image_compressor,
        " Image shape: ",
        image_shape,
    )
    #image_compressor = image_metadata[".zarray"]["compressor"]
    
    # Add this check and conversion:
    if isinstance(image_compressor, dict):
        if image_compressor["id"] == "blosc":
            image_compressor = numcodecs.Blosc(
                cname=image_compressor.get("cname", "zstd"),
                clevel=image_compressor.get("clevel", 1),
                shuffle=image_compressor.get("shuffle", 1),
                blocksize=image_compressor.get("blocksize", 0)
            )
        else:
            # Default fallback if compressor type is unknown
            image_compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    elif image_compressor is None:
        image_compressor = numcodecs.Blosc(cname="zstd", clevel=3)

    #print("Image compressor: ", image_compressor)

    #image_compressor = (
    #    numcodecs.Blosc(cname="zstd", clevel=3) if image_compressor is None else image_compressor
    #)
    print("Image compressor: ", image_compressor)
    # Reading segmentation mask
    seg_mask_reader = ImageReaderFactory().create(
        data_path=segmentation_mask_path,
        parse_path=False,
        multiscale=".",
    )

    lazy_mask_data = extract_data(seg_mask_reader.as_dask_array())

    output_filepath = output_folder.joinpath(filename).as_posix()
    output_params = {
        "chunksize": [1, 1, 128, 128, 128],
        "resolution_zyx": resolution_zyx,
        "dtype": image_lazy_data.dtype,
        "path": output_filepath,
        "compressor": image_compressor,
        "dimension_separator": "/",
    }

    # Upsampling the segmentation mask (creates level 0)
    print("\n" + "="*60)
    print("STEP 1: Upsampling segmentation mask")
    print("="*60)
    # Upsampling the segmentation mask
    upscale_zarr_with_padding(
        input_zarr=lazy_mask_data,
        output_params=output_params,
        upscale_factors_zyx=(1, 4, 4),  # TODO Calculate factors based on metadata
        new_shape=image_lazy_data.shape,
        n_workers=n_workers,
    )

    generate_pyramid = True
    # Generate pyramid if requested
    if generate_pyramid:
        print("\n" + "="*60)
        print("STEP 2: Generating multiscale pyramid")
        print("="*60)
        generate_multiscale_pyramid(
            zarr_path=output_filepath,
            max_level=max_pyramid_level,
            downsample_factor=2,
            n_workers=n_workers,
            chunk_size=output_params["chunksize"],
        )
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Output location: {output_filepath}")
    if generate_pyramid:
        print(f"Pyramid levels: 0 (full resolution) through {max_pyramid_level}")
    print("="*60 + "\n")
