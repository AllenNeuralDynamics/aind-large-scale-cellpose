""" top level run script """

import os
from pathlib import Path
import argparse 

from aind_large_scale_cellpose.segment import segment
from aind_large_scale_cellpose.cellpose_segmentation.utils import utils

def run():
    """Runs large-scale cell segmentation"""
    # Code ocean folders
    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")
    scratch_folder = os.path.abspath("../scratch")

    # NOTE: Change the cell diameter based on multiscale
    multiscale = "2"

    # Cellpose params
    cellpose_params = {
        "model_name": "cyto",  # "../data/CP_20240905_144444_LC",
        "cell_diameter": 30,
        "min_cell_volume": 95,
        "percentile_range": (10, 99),
        "flow_threshold": 0.0,
    }

    scheduler_params = {
        "target_size_mb": 3072,
        "n_workers": 0,
        "predict_gradients": {
            "slices_per_axis": [20, 20, 20],
            "output_gradients_path": f"{scratch_folder}/gradients.zarr",
        },
        "combine_gradients": {
            "prediction_chunksize": (3, 3, 128, 128, 128),
            "super_chunksize": (3, 3, 128, 128, 128),
            "n_workers": 0,
            "output_combined_gradients_path": f"{scratch_folder}/combined_gradients.zarr",
            "output_cellprob_path": f"{scratch_folder}/combined_cellprob.zarr",
        },
        "flow_centroids": {
            "output_flows": f"{scratch_folder}/pflows.zarr",
            "output_hists": f"{scratch_folder}/hists.zarr",
            "prediction_chunksize": (3, 128, 128, 128),
        },
        "generate_masks": {
            "output_mask": f"{results_folder}/segmentation_mask_orig_res.zarr",
            "prediction_chunksize": (3, 128, 128, 128),
            "super_chunksize": (3, 512, 512, 512),
        },
    }

    processing_manifest_path = Path(data_folder).joinpath(
        "processing_manifest.json"
    )

    if not processing_manifest_path.exists():
        raise FileNotFoundError(f"Path {processing_manifest_path} not found!")

    processing_manifest = utils.read_json_as_dict(filepath=processing_manifest_path)
    segmentation_channels = processing_manifest.get("segmentation_channels", None)
    
    if segmentation_channels is None:
        raise ValueError(f"Please, provide segmentation channels in manifest. {processing_manifest}")

    dataset_paths = []
    background_channel_number = segmentation_channels.get("background", None)
    nuclei_channel_number = segmentation_channels.get("nuclear", None)

    if background_channel_number is None:
        raise ValueError("Background channel is necessary for segmentation.")
    
    # Will explicitly fail if the path does not exist
    background_channel = list(Path(data_folder).glob(f"*{background_channel_number}.ome.zarr"))[0]
    dataset_paths.append(background_channel)

    nuclei_channel = None
    if nuclei_channel_number is None:
        msg = "Nuclei channel not provided, please check parameters if it's an error!"
        print(msg)
    
    else:
        nuclei_channel = list(Path(data_folder).glob(f"*{nuclei_channel_number}.ome.zarr"))[0]
        dataset_paths.append(nuclei_channel)

    print(f'Segmenting with channels: {dataset_paths}')

    segment(
        dataset_paths=dataset_paths,
        multiscale=multiscale,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
        global_normalization=True,
        cellpose_params=cellpose_params,
        scheduler_params=scheduler_params,
        code_ocean=True,
        upsample_masks_levels=1,
    )

if __name__ == "__main__":
    run()
