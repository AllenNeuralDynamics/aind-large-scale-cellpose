"""top level run script"""

import argparse
import os

from aind_large_scale_cellpose.segment import segment


def run(dataset):
    """Runs large-scale cell segmentation

    Args:
        dataset (str): Name of the dataset to process
    """
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

    # BKG_CHN = 'fused/channel_405.zarr'
    # NUCLEI_CHN = 'fused/channel_594.zarr'

    # single tile BKG_CHN
    BKG_CHN = "SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr"

    background_channel = f"{data_folder}/{dataset}/{BKG_CHN}"
    # nuclei_channel = f"{data_folder}/{dataset}/{NUCLEI_CHN}"
    dataset_paths = [background_channel]  # , nuclei_channel]

    print(f"Segmenting {dataset_paths}")

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
    parser = argparse.ArgumentParser(description="Run cell segmentation on a specified dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset to process")
    args = parser.parse_args()
    run(args.dataset)
