# aind-large-scale-cellpose

Large-scale 3D cell segmentation using [Cellpose](https://github.com/MouseLand/cellpose), designed for whole-brain light-sheet microscopy datasets stored in OME-Zarr format. The pipeline processes data in overlapping chunks to handle volumes that are too large to fit in memory, producing a single globally-consistent segmentation mask.

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration reference](#configuration-reference)
- [Upscaling segmentation masks](#upscaling-segmentation-masks)
- [Disk space requirements](#disk-space-requirements)
- [Contributing](#contributing)

---

## How it works

The pipeline runs five sequential stages, each writing intermediate results to disk as Zarr datasets:

1. **Gradient prediction** — The dataset is divided into overlapping 3D chunks. Cellpose predicts spatial gradients along YX, ZX, and ZY planes for each chunk. Data is normalized globally (using precomputed percentiles across the whole volume) or locally (per-plane) before inference.

2. **Gradient combination** — The per-plane gradient predictions from step 1 are combined into a single ZYX gradient volume and a cell probability map. No overlap is required for this step.

3. **Flow integration and centroid detection** — ZYX flows are integrated in overlapping chunks to produce per-cell histograms and seed centroids. The overlap on each axis must be at least `2 × mean_cell_diameter` to prevent flow cut-offs at chunk boundaries.

4. **Segmentation mask generation** — Global cell centroids from step 3 are used to assign unique cell IDs. Each cell ID is propagated through the ZYX flow field in overlapping chunks to produce the final segmentation mask. Chunk overlap must match step 3.

5. **Mask upscaling** (optional) — If the segmentation was run on a downsampled pyramid level, the mask can be upscaled back to full resolution. Per-axis scale factors are derived automatically from the OME-Zarr coordinate transformation metadata, correctly handling anisotropic datasets where Z and XY have different downsampling factors per level.

---

## Installation

Requires Python >= 3.9.

```bash
pip install aind-large-scale-cellpose
```

For development, install with optional dev dependencies:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-large-scale-cellpose.git
cd aind-large-scale-cellpose
pip install -e ".[dev]"
```

---

## Quick start

The main entry point is `aind_large_scale_cellpose.segment.segment`. See [code/run_capsule.py](code/run_capsule.py) for a complete working example. A minimal call looks like:

```python
from aind_large_scale_cellpose.segment import segment

segment(
    dataset_paths=["path/to/image.zarr"],
    multiscale="2",          # pyramid level to run segmentation at
    results_folder="results",
    scratch_folder="scratch",
    cellpose_params={
        "model_name": "cyto",
        "cell_diameter": 30,
        "min_cell_volume": 95,
        "percentile_range": (10, 99),
        "flow_threshold": 0.0,
    },
    scheduler_params={
        "target_size_mb": 3072,
        "n_workers": 0,
        "predict_gradients": {
            "slices_per_axis": [20, 20, 20],
            "output_gradients_path": "scratch/gradients.zarr",
        },
        "combine_gradients": {
            "prediction_chunksize": (3, 3, 128, 128, 128),
            "super_chunksize": (3, 3, 128, 128, 128),
            "n_workers": 0,
            "output_combined_gradients_path": "scratch/combined_gradients.zarr",
            "output_cellprob_path": "scratch/combined_cellprob.zarr",
        },
        "flow_centroids": {
            "output_flows": "scratch/pflows.zarr",
            "output_hists": "scratch/hists.zarr",
            "prediction_chunksize": (3, 128, 128, 128),
        },
        "generate_masks": {
            "output_mask": "results/segmentation_mask_orig_res.zarr",
            "prediction_chunksize": (3, 128, 128, 128),
            "super_chunksize": (3, 512, 512, 512),
        },
    },
    global_normalization=True,
    upsample_masks_levels=1,
)
```

The input dataset must be an OME-Zarr multiscale store. Pass multiple paths to `dataset_paths` for multi-channel segmentation (e.g., background + nuclei channels); the first channel is assumed to be the background.

---

## Configuration reference

### `cellpose_params`

| Key | Type | Description |
|-----|------|-------------|
| `model_name` | `str` | Cellpose model to use (`"cyto"`, `"nuclei"`, or a path to a custom model). |
| `cell_diameter` | `int` | Expected mean cell diameter in pixels **at the chosen pyramid level**. This controls the overlap between chunks — increase it if cells are cut off at boundaries. |
| `min_cell_volume` | `int` | Minimum cell volume in voxels. Cells smaller than this are discarded. Corresponds to `min_size` in Cellpose. |
| `percentile_range` | `tuple[int, int]` | (low, high) percentiles used for global intensity normalization, e.g. `(10, 99)`. |
| `flow_threshold` | `float` | Cellpose flow error threshold. Lower values are more permissive. `0.0` disables the threshold. |

### `scheduler_params`

| Key | Type | Description |
|-----|------|-------------|
| `target_size_mb` | `int` | Target chunk size in MB when loading data for gradient prediction. |
| `n_workers` | `int` | Number of Dask workers for gradient prediction. `0` uses the default scheduler. |

#### `predict_gradients`

| Key | Type | Description |
|-----|------|-------------|
| `slices_per_axis` | `list[int]` | Number of slices per axis `[z, y, x]` used to divide the volume for gradient prediction. |
| `output_gradients_path` | `str` | Path for the per-axis gradient Zarr output. Can be deleted after the run. |

#### `combine_gradients`

| Key | Type | Description |
|-----|------|-------------|
| `prediction_chunksize` | `tuple` | Chunk size `(gradients, c, z, y, x)` for the combination step. |
| `super_chunksize` | `tuple` | Super-chunk size for batched reads. |
| `n_workers` | `int` | Number of Dask workers. |
| `output_combined_gradients_path` | `str` | Path for the combined ZYX gradient Zarr. Can be deleted after the run. |
| `output_cellprob_path` | `str` | Path for the cell probability Zarr. Can be deleted after the run. |

#### `flow_centroids`

| Key | Type | Description |
|-----|------|-------------|
| `output_flows` | `str` | Path for the integrated ZYX flow Zarr. Can be deleted after the run. |
| `output_hists` | `str` | Path for the cell histogram Zarr. Can be deleted after the run. |
| `prediction_chunksize` | `tuple` | Chunk size `(z, y, x)` including overlap. Must satisfy: `chunk ≥ core + 2 × cell_diameter` on each axis. |

#### `generate_masks`

| Key | Type | Description |
|-----|------|-------------|
| `output_mask` | `str` | Path for the segmentation mask Zarr at the segmented pyramid level. |
| `prediction_chunksize` | `tuple` | Chunk size matching the one used in `flow_centroids`. |
| `super_chunksize` | `tuple` | Super-chunk size for batched reads during mask generation. |

---

## Upscaling segmentation masks

When segmentation is run on a downsampled pyramid level (e.g., `multiscale="2"`), pass `upsample_masks_levels` to upscale the mask back toward full resolution:

- `upsample_masks_levels=1` — upscale to full resolution (level 0) only, no pyramid.
- `upsample_masks_levels=N` (N > 1) — upscale to level 0 and write an N-level OME-Zarr pyramid.

Per-axis upscale factors are computed automatically from the OME-Zarr coordinate transformation metadata (`source_multiscale → dest_multiscale="0"`). This correctly handles anisotropic datasets where Z and XY axes have different scale factors per pyramid level (e.g., SmartSPIM Z1).

---

## Disk space requirements

The pipeline requires temporary disk space for several intermediate Zarr datasets in the scratch folder. As a rough estimate, the total scratch space needed is about **3–5× the size of the input volume at the chosen pyramid level**. Only `segmentation_mask.zarr` in the results folder needs to be kept permanently; all intermediate datasets in the scratch folder can be deleted once the run completes.

The chunk sizes in `scheduler_params` directly affect peak memory usage per worker. If a worker runs out of memory, reduce `target_size_mb` or the spatial dimensions of `prediction_chunksize`.

---

## Contributing

To develop the code, install the packages described in the Dockerfile or via `pip install -e ".[dev]"`.

### Linters and testing

- Run tests with coverage:

```bash
coverage run -m unittest discover && coverage report
```

- Check docstring coverage:

```bash
interrogate .
```

- Check code style:

```bash
flake8 . --max-line-length=100
```

- Auto-format code:

```bash
black .
```

- Sort imports:

```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style commit messages:

```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected and type (mandatory) is one of:

| Type | When to use |
|------|-------------|
| `build` | Changes to the build system or external dependencies |
| `ci` | Changes to CI configuration files and scripts |
| `docs` | Documentation only changes |
| `feat` | A new feature |
| `fix` | A bug fix |
| `perf` | A code change that improves performance |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `test` | Adding missing tests or correcting existing tests |
