"""
Create segmentation masks from raw segmentation masks.
"""

import os
from pathlib import Path

import dask.array as da
from aind_large_scale_cellpose.cellpose_segmentation.upscale_masks import upscale_mask
from aind_large_scale_cellpose.cellpose_segmentation.utils import utils


def main():
    """ Main function """
    bucket = ""

    datasets = []

    prefix = "cell_body_segmentation"
    raw_data = "image_tile_fusing/fused/channel_405.zarr"

    file = "segmentation_mask_orig_res.zarr"
    multiscale = "2"

    co_cpus = 5

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        dataset_root = f"s3://{bucket}/{dataset}"

        path_to_upsample = f"{dataset_root}/{prefix}/{file}"

        raw_data_path = f"{dataset_root}/{raw_data}"

        results_folder = f"/results/{dataset}"
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        print(f"Loading segmentation mask: {path_to_upsample}")
        lazy_mask_data = da.from_zarr(path_to_upsample)

        print("Upscaling segmentation mask...")

        # source_multiscale is the pyramid level the segmentation was run at.
        # dest_multiscale="0" corresponds to full resolution.
        # Per-axis upscale factors are derived automatically from
        # OME-Zarr coordinate transformation metadata, correctly handling
        # anisotropic datasets where Z and XY differ.
        resolution_zyx, _, per_level_scale_factors = upscale_mask.upscale_mask(
            dataset_path=raw_data_path,
            mask_data=lazy_mask_data,
            output_folder=results_folder,
            filename="segmentation_mask.zarr",
            dest_multiscale="0",
            source_multiscale=multiscale,
            n_workers=co_cpus,
        )

        # Create multiscale pyramid metadata based on the original image's
        # OME-Zarr metadata, correctly handling anisotropic pyramids.
        output_upscaled_mask = str(Path(results_folder) / "segmentation_mask.zarr")

        upscale_mask.write_multiscales(
            path_to_data=output_upscaled_mask,
            voxel_size=list(resolution_zyx),
            scale_factors_per_level=per_level_scale_factors,
        )

        print(f"Finished processing: {dataset}")


if __name__ == "__main__":
    main()
