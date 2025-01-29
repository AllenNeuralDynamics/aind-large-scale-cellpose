import os
import json
import pathlib
from pathlib import Path
import argparse 
import s3fs

from aind_large_scale_cellpose.segment import segment

def read_processing_manifest(DATA_FOLDER):
    """Load the processing manifest JSON file"""
    manifest_path = list(Path(DATA_FOLDER).glob("derived/processing_manifest.json"))
    
    if not len(manifest_path):
        print('Didn\'t find pipeline processing manifest')
        manifest_path = list(Path(DATA_FOLDER).glob("*/derived/processing_manifest.json"))
        if not len(manifest_path):
            raise FileNotFoundError("No capsule processing_manifest.json was found!")

    print(f'Manifest_path {manifest_path}')

    try:
        with open(manifest_path[0], 'r') as f:
            manifest = json.load(f)
            print(f"Loaded manifest with channels: {manifest.get('spot_channels', [])}")
    except FileNotFoundError:
        manifest = None
        raise FileNotFoundError(f"Processing manifest not found at {manifest_path}")
    
    return manifest

def find_zarr_file(data_folder, dataset, channel):
    """Find zarr file in either fused or image_tile_fusing/OMEZarr directories
    
    Args:
        data_folder (str): Base data folder path
        dataset (str): Dataset name
        channel (str): Channel file name
    
    Returns:
        str: Full path to zarr file if found, None otherwise
    """
    # Check image_tile_fusing/OMEZarr first
    omezarr_path = os.path.join(data_folder, dataset, "image_tile_fusing/OMEZarr", channel)
    if os.path.exists(omezarr_path):
        return omezarr_path
        
    # Check fused directory as backup
    fused_path = os.path.join(data_folder, dataset, "fused", channel)
    if os.path.exists(fused_path):
        return fused_path
        
    return None

def upload_results_to_s3(local_results_folder, s3_path):
    """Upload results to S3 using s3fs
    
    Args:
        local_results_folder (str): Local folder containing results
        s3_path (str): S3 destination path
    """
    try:
        # Initialize S3 filesystem
        s3 = s3fs.S3FileSystem(anon=False)  # Uses AWS credentials from environment
        
        print(f"Starting upload to {s3_path}")
        
        # Recursively upload the results folder
        s3.put(local_results_folder, s3_path, recursive=True)
        
        print(f"Successfully uploaded results to {s3_path}")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise

def run(dataset):
    """Runs large-scale cell segmentation
    
    Args:
        dataset (str): Name of the dataset to process
    """
    # Code ocean folders
    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")
    scratch_folder = os.path.abspath("../scratch")

    # Read processing manifest
    manifest = read_processing_manifest(data_folder)
    
    # Check if this is round 1 - only process if it is
    if manifest.get('round') != 1:
        print(f"Skipping segmentation for round {manifest.get('round')} - only processing round 1")
        return

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

    # Get segmentation channels from manifest, with defaults
    background_channel = manifest.get('background_channel')
    nuclei_channel = manifest.get('nuclei_channel')

    if not background_channel and not nuclei_channel:
        # Default to channel_405 if no channels specified
        background_zarr = find_zarr_file(data_folder, dataset, "channel_405.zarr")
        if not background_zarr:
            raise FileNotFoundError("Could not find default channel_405.zarr in any expected location")
        dataset_paths = [background_zarr]
    else:
        # Use specified channels
        dataset_paths = []
        if background_channel:
            background_zarr = find_zarr_file(data_folder, dataset, background_channel)
            if background_zarr:
                dataset_paths.append(background_zarr)
            else:
                print(f"Warning: Could not find background channel {background_channel}")
                
        if nuclei_channel:
            nuclei_zarr = find_zarr_file(data_folder, dataset, nuclei_channel)
            if nuclei_zarr:
                dataset_paths.append(nuclei_zarr)
            else:
                print(f"Warning: Could not find nuclei channel {nuclei_channel}")

    if not dataset_paths:
        raise FileNotFoundError("No valid zarr files found for segmentation")

    print(f'Segmenting {dataset_paths}')

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

    # Upload results to S3 if location specified
    s3_location = manifest.get('s3_location')
    if s3_location:
        s3_results_path = f"{s3_location.rstrip('/')}/cell_body_segmentation"
        upload_results_to_s3(results_folder, s3_results_path)

def debug(): 
    dataset = 'HCR_732195-ROI2-cell1_2024-06-15_06-00-00'
    run(dataset)
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Run cell segmentation on a specified dataset")
    #parser.add_argument("dataset", type=str, help="Name of the dataset to process")
    #args = parser.parse_args()
    #run(args.dataset)
    debug()