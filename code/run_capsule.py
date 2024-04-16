""" top level run script """

import os

from cell_segmentation.segment import segment


def run():
    """Runs large-scale cell segmentation"""
    # Code ocean folders
    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")
    scratch_folder = os.path.abspath("../data")

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
        "cell_diameter": 15,
        "min_cell_volume": 95,
        "percentile_range": (10, 99),
        "flow_threshold": 0.0,
    }

    # dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    background_channel = f"{data_folder}/{IMAGE_PATH}/{BKG_CHN}"
    nuclei_channel = f"{data_folder}/{IMAGE_PATH}/{NUCLEI_CHN}"

    dataset_paths = [background_channel, nuclei_channel]

    segment(
        dataset_paths=dataset_paths,
        multiscale=multiscale,
        results_folder=results_folder,
        data_folder=data_folder,
        scratch_folder=scratch_folder,
        global_normalization=True,
        cellpose_params=cellpose_params,
    )


if __name__ == "__main__":
    run()
