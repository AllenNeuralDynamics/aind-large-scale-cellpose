""" top level run script """

import os

from aind_large_scale_cellpose.segment import segment


def run():
    """Runs large-scale cell segmentation"""
    # Code ocean folders
    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")
    scratch_folder = os.path.abspath("../scratch")

    # Dataset to process
    IMAGE_PATH = "HCR_BL6-000_2023-06-1_00-00-00_fused_2024-04-02_20-06-14"
    # "HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49"
    BKG_CHN = "channel_405.zarr"
    NUCLEI_CHN = "channel_3.zarr"

    # NOTE: Change the cell diameter based on multiscale
    multiscale = "2"

    # Cellpose params
    cellpose_params = {
        "model_name": "cyto",
        "cell_diameter": 30,
        "min_cell_volume": 95,
        "percentile_range": (10, 99),
        "flow_threshold": 0.0,
    }

    scheduler_params = {
        "target_size_mb": 3072,
        "n_workers": 0,
        "predict_gradients": {
            "slices_per_axis": [48, 48, 45],
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
            "output_mask": f"{results_folder}/segmentation_mask.zarr",
            "prediction_chunksize": (3, 128, 128, 128),
            "super_chunksize": (3, 512, 512, 512),
        },
    }

    # dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    background_channel = f"{data_folder}/{IMAGE_PATH}/{BKG_CHN}"
    nuclei_channel = f"{data_folder}/{IMAGE_PATH}/{NUCLEI_CHN}"

    dataset_paths = [background_channel, nuclei_channel]

    segment(
        dataset_paths=dataset_paths,
        multiscale=multiscale,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
        global_normalization=True,
        cellpose_params=cellpose_params,
        scheduler_params=scheduler_params,
        code_ocean=True,
    )


if __name__ == "__main__":
    run()
