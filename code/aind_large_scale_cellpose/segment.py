"""
Main file to run segmentation
"""

import os
from typing import Dict, List, Optional

from .cellpose_segmentation._shared.types import PathLike
from .cellpose_segmentation.combine_gradients import combine_gradients
from .cellpose_segmentation.compute_flows import generate_flows_and_centroids
from .cellpose_segmentation.compute_masks import generate_masks
from .cellpose_segmentation.predict_gradients import predict_gradients


def segment(
    dataset_paths: List[PathLike],
    multiscale: str,
    results_folder: PathLike,
    scratch_folder: PathLike,
    cellpose_params: Dict,
    scheduler_params: Dict,
    global_normalization: Optional[bool] = True,
    code_ocean: Optional[bool] = True,
):
    """
    Segments a Z1 dataset.

    Parameters
    ----------
    dataset_paths: List[PathLike]
        Paths where the datasets in Zarr format are located.
        These could be background channel and nuclei channel in
        as different zarr datasets.
        If the data is in the cloud, please provide the
        path to it. E.g., s3://bucket-name/path/image.zarr

    multiscale: str
        Dataset multiscale.

    results_folder: PathLike
        Path of the results folder.

    scratch_folder: PathLike
        Path of the scratch folder.

    cellpose_params: Dict
        Cellpose parameters

    scheduler_params: Dict
        Scheduler params. Setting these parameters
        correctly will allows proper segmentation and
        faster processing times.

    global_normalization: Optional[bool]
        True if we want to compute the normalization
        based on the whole dataset. Default: True

    code_ocean: Optional[bool]
        If the instance is running in a code ocean environment.

    """
    len_datasets = len(dataset_paths)

    if not len_datasets:
        ValueError("Please, provide valid paths. Empty list!")

    # Validating output folder
    if len_datasets and os.path.exists(results_folder):

        # Data loader params
        super_chunksize = None
        target_size_mb = scheduler_params["target_size_mb"]
        n_workers = scheduler_params["n_workers"]
        batch_size = 1

        # Cellpose parameters
        model_name = cellpose_params["model_name"]
        cell_diameter = cellpose_params["cell_diameter"]
        min_cell_volume = cellpose_params["min_cell_volume"]
        percentile_range = cellpose_params["percentile_range"]
        flow_threshold = cellpose_params["flow_threshold"]

        # channels for segmentation, we assume background channel is in 0, nuclei 1, ...
        cell_channels = (
            [0, 0] if len(dataset_paths) == 1 else [i for i in range(len(dataset_paths))]
        )

        # Large-scale prediction of gradients
        slices_per_axis = scheduler_params["predict_gradients"]["slices_per_axis"]
        output_gradients_path = scheduler_params["predict_gradients"]["output_gradients_path"]

        dataset_shape = predict_gradients(
            dataset_paths=dataset_paths,
            multiscale=multiscale,
            output_gradients_path=output_gradients_path,
            slices_per_axis=slices_per_axis,
            target_size_mb=target_size_mb,
            n_workers=n_workers,
            batch_size=batch_size,
            super_chunksize=super_chunksize,
            global_normalization=global_normalization,
            model_name=model_name,
            cell_diameter=cell_diameter,
            results_folder=results_folder,
            scratch_folder=scratch_folder,
            cell_channels=cell_channels,
            min_cell_volume=min_cell_volume,
            percentile_range=percentile_range,
            code_ocean=code_ocean,
        )

        # Large-scale combination of predicted gradients
        prediction_chunksize = scheduler_params["combine_gradients"]["prediction_chunksize"]
        super_chunksize = scheduler_params["combine_gradients"]["super_chunksize"]
        n_workers = scheduler_params["combine_gradients"]["n_workers"]  # 0

        # output combined gradients path and cell probabilities
        output_combined_gradients_path = scheduler_params["combine_gradients"][
            "output_combined_gradients_path"
        ]
        output_cellprob_path = scheduler_params["combine_gradients"]["output_cellprob_path"]

        combine_gradients(
            dataset_path=output_gradients_path,
            multiscale=".",
            output_combined_gradients_path=output_combined_gradients_path,
            output_cellprob_path=output_cellprob_path,
            prediction_chunksize=prediction_chunksize,
            super_chunksize=super_chunksize,
            target_size_mb=None,
            n_workers=n_workers,
            batch_size=1,
            results_folder=results_folder,
        )

        # Output paths
        output_combined_pflows = scheduler_params["flow_centroids"]["output_flows"]
        output_combined_hists = scheduler_params["flow_centroids"]["output_hists"]
        prediction_chunksize = scheduler_params["flow_centroids"]["prediction_chunksize"]

        output_combined_pflows = f"{results_folder}/pflows.zarr"
        output_combined_hists = f"{results_folder}/hists.zarr"

        # Large-scale generation of flows, centroids and hists
        cell_centroids_path = generate_flows_and_centroids(
            dataset_path=output_combined_gradients_path,
            output_pflow_path=output_combined_pflows,
            output_hist_path=output_combined_hists,
            multiscale=".",
            axis_overlap=cell_diameter // 2,  # Used to get the overlapping area
            prediction_chunksize=prediction_chunksize,
            target_size_mb=target_size_mb,
            n_workers=n_workers,
            batch_size=batch_size,
            super_chunksize=None,
            results_folder=results_folder,
        )

        # Output mask
        output_segmentation_mask = scheduler_params["generate_masks"]["output_mask"]
        prediction_chunksize = scheduler_params["generate_masks"]["prediction_chunksize"]
        super_chunksize = scheduler_params["generate_masks"]["super_chunksize"]

        # Large-scale segmentation mask generation
        generate_masks(
            dataset_path=output_combined_pflows,
            multiscale=".",
            hists_path=output_combined_hists,
            cell_centroids_path=cell_centroids_path,
            output_seg_mask_path=output_segmentation_mask,
            original_dataset_shape=dataset_shape,
            axis_overlap=cell_diameter // 2,  # Used to get the overlapping area
            prediction_chunksize=prediction_chunksize,
            target_size_mb=None,
            n_workers=n_workers,
            batch_size=batch_size,
            super_chunksize=super_chunksize,
            min_cell_volume=min_cell_volume,
            flow_threshold=flow_threshold,
            results_folder=results_folder,
        )

    else:
        print("Provided paths do not exist!")


if __name__ == "__main__":
    segment()
