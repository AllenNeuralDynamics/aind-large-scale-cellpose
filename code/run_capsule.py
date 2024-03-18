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
    IMAGE_PATH = "HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49"
    TILE_NAME = "channel_405.zarr"
    # dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    dataset_path = f"{data_folder}/{IMAGE_PATH}/{TILE_NAME}"

    segment(
        dataset_path=dataset_path,
        multiscale="2",
        results_folder=results_folder,
        data_folder=data_folder,
        scratch_folder=scratch_folder,
    )


if __name__ == "__main__":
    run()
