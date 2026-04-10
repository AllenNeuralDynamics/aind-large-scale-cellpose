"""
Functions to generate the OMEZarr 0.5 metadata
"""

from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from ome_zarr.format import CurrentFormat
from ome_zarr.writer import Format, _get_valid_axes, _validate_datasets


def _get_pyramid_metadata():
    """
    Gets the image pyramid metadata
    using tensorstore package
    """
    return {
        "metadata": {
            "description": "Downscaling with xarray multiscale",
            "method": "xarray.reducers.windowed_mode_countless",
            "version": "2.1.0",
            "args": "[false]",
            # No extra parameters were used different
            # from the orig. array and scales
            "kwargs": {},
        }
    }


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [f"Channel:{image_name}:{i}" for i in range(data_shape[1])]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    if channel_startend is None:
        channel_startend = channel_minmax

    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
                },
            }
        )

    ome = {
        "channels": ch,
    }
    return ome


def _compute_scales(
    scale_num_levels: int,
    scale_factors_per_level: List[List[float]],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translations: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """
    Generate the list of coordinate transformations
    and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factors_per_level: per-level scale factors in ZYX order, e.g.
        [[1, 2, 2], [2, 2, 2]]. If fewer entries than levels, the last
        entry is reused for remaining levels.
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each
    chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset
    in physical units in each dimension

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[-3],
                    pixelsizes[-2],
                    pixelsizes[-1],
                ],
            }
        ]
    ]
    if translations is not None:
        transforms[0].append({"type": "translation", "translation": translations[0]})
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            level_factor = scale_factors_per_level[min(i, len(scale_factors_per_level) - 1)]
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * level_factor[-3],
                            last_scale[3] * level_factor[-2],
                            last_scale[4] * level_factor[-1],
                        ],
                    }
                ]
            )
            if translations is not None:
                transforms[-1].append({"type": "translation", "translation": translations[i + 1]})
            lastz = int(np.ceil(lastz / level_factor[-3]))
            lasty = int(np.ceil(lasty / level_factor[-2]))
            lastx = int(np.ceil(lastx / level_factor[-1]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(time_unit: str = "millisecond", space_unit: str = "micrometer") -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def _validate_axes_for_format(axes: List[Dict], fmt: Format):
    """
    Validate and adjust axes based on the format version.

    Parameters
    ----------
    axes : list of str or list of dict or None
        The axes provided by the user.
    fmt : Format
        The OME-Zarr format version.

    Returns
    -------
    tuple
        A tuple of (validated_axes, ndim), where:
        - validated_axes is the processed axes or None
        - ndim is the number of dimensions, or -1 if axes is None
    """
    if axes is not None:
        if fmt.version in ("0.1", "0.2"):
            print("axes ignored for version 0.1 or 0.2")
            return None, -1
        else:
            axes = _get_valid_axes(axes=axes, fmt=fmt)
            return axes, len(axes) if axes is not None else -1
    return None, -1


def _validate_omero_metadata(omero_metadata: Dict):
    """
    Validate the structure and types of OMERO metadata.

    Parameters
    ----------
    omero_metadata : dict or None
        OMERO metadata dictionary containing channel information.

    Raises
    ------
    TypeError
        If color is not a 6-character hex string, or if window is not a dict,
        or if any window value is not an int or float.
    KeyError
        If a required key is missing from the window dictionary.
    """
    if not omero_metadata:
        return
    for c in omero_metadata["channels"]:
        if "color" in c:
            if not isinstance(c["color"], str) or len(c["color"]) != 6:
                raise TypeError("`'color'` must be a hex code string.")
        if "window" in c:
            if not isinstance(c["window"], dict):
                raise TypeError("`'window'` must be a dict.")
            for p in ["min", "max", "start", "end"]:
                if p not in c["window"]:
                    raise KeyError(f"`'{p}'` not found in `'window'`.")
                if not isinstance(c["window"][p], (int, float)):
                    raise TypeError(f"`'{p}'` must be an int or float.")


def add_multiscales_metadata(
    group: Dict,
    datasets: list[dict],
    fmt: Format = CurrentFormat(),
    axes: List[Dict] = None,
    name=None,
    omero_metadata=None,
    **metadata,
) -> Dict:
    """
    Write multiscales metadata into a Zarr group.

    This function writes metadata in the OME-NGFF multiscales format to the
    specified group dictionary. It validates dataset transforms, axes,
    and OMERO metadata, and adds the required multiscales and optional
    OMERO entries.

    Parameters
    ----------
    group : dict
        Dictionary representing a Zarr group, where metadata will be written.
    datasets : list of dict
        Each dict should contain 'path' and 'coordinateTransformations'.
    fmt : Format, optional
        The version of the OME-Zarr format to use.
        Defaults to the most current.
    axes : list of str or list of dict, optional
        The axes specification. Required for format version 0.3 and above.
    name : str, optional
        Optional name for the multiscale image. If not provided,
        falls back to group name.
    omero_metadata : dict, optional
        Dictionary containing OMERO metadata such as channels and windows.
    **metadata : str or dict or list of dict
        Additional metadata to include in the multiscales entry.

    Returns
    -------
    Dict:
        Dictionary with the omezarr metadata
    """
    axes, ndim = _validate_axes_for_format(axes, fmt)
    _validate_omero_metadata(omero_metadata)

    multiscales = [
        dict(
            name=name or group.get("name"),
            axes=axes,
            datasets=_validate_datasets(datasets, ndim, fmt),
            version="0.4",
            **metadata,
        )
    ]

    # group.setdefault("attributes", {}).setdefault("ome", {})
    # group["attributes"]["ome"]["multiscales"] = multiscales
    # group["attributes"]["ome"]["omero"] = omero_metadata
    group["multiscales"] = multiscales
    group["omero"] = omero_metadata

    return group


def _downscale_origin(
    array_shape: List[int],
    origin: List[float],
    voxel_size: List[float],
    scale_factors_per_level: List[List[int]],
    n_levels: int,
):
    """
    Calculate new origins for downscaled coordinate grids.

    Parameters
    ----------
    array_shape : List[int]
        Shape of the array in [t, c, z, y, x] order.
    origin : list or tuple of float
        The initial origin coordinates (z, y, x) of the array.
    voxel_size : list or tuple of float
        The size of each voxel along the (z, y, x) dimensions.
    scale_factors_per_level : List[List[int]]
        Per-level scale factors in ZYX order, e.g. [[1, 2, 2], [2, 2, 2]].
        If fewer entries than levels, the last entry is reused.
    n_levels : int
        The number of downscaling levels to calculate.

    Returns
    -------
    new_origins : list of list of float
        A list of new origin coordinates for each downscaled level.
    """
    current_shape = np.array(array_shape[-3:], dtype=np.int32)
    current_origin = np.array(origin[-3:], dtype=np.float64)
    current_voxel_size = np.array(voxel_size[-3:], dtype=np.float64)

    new_origins = [current_origin.tolist()]
    for i in range(n_levels - 1):
        level_factors = np.array(
            scale_factors_per_level[min(i, len(scale_factors_per_level) - 1)][-3:],
            dtype=np.int32,
        )

        # Calculate the center shift for the new origin
        center_shift = (current_voxel_size * (level_factors - 1)) / 2
        current_origin += center_shift

        current_shape = np.ceil(current_shape / level_factors).astype(int)
        next_voxel_size = current_voxel_size * level_factors
        current_voxel_size = next_voxel_size

        # Append the new origin
        new_origins.append([0.0, 0.0] + current_origin.tolist())

    # Ensure the initial origin is 5D
    if len(new_origins[0]) < 5:
        new_origins[0] = [0.0, 0.0] + new_origins[0]

    return new_origins


def write_ome_ngff_metadata(
    arr_shape: List[int],
    chunk_size: List[int],
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    origin: list = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    arr_shape : List[int]
        List of ints with the dataset shape.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : List[List[int]]
        Per-level scale factors in ZYX order, e.g. [[1, 2, 2], [2, 2, 2]].
        If fewer entries than levels, the last entry is reused.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    group = dict(
        # zarr_format=3,
        # node_type="group",
        # attributes={},
    )

    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    omero_metadata = _build_ome(
        arr_shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    axes_5d = _get_axes_5d()

    if origin is not None:
        origin = _downscale_origin(arr_shape, origin[-3:], voxel_size[-3:], scale_factors, n_lvls)

    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, chunk_size, arr_shape, origin
    )
    fmt.validate_coordinate_transformations(len(arr_shape), n_lvls, coordinate_transformations)
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # group["attributes"]["ome"] = {"version": "0.4"}

    # Writing the multiscale metadata
    group = add_multiscales_metadata(
        group=group,
        datasets=datasets,
        fmt=fmt,
        axes=axes_5d,
        name=image_name,
        omero_metadata=omero_metadata,
        **metadata,
    )

    return group
