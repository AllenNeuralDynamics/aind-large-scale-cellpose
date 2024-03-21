# aind-z1-cell-segmentation

Large-scale cell segmentation using cellpose for Z1 data.

The approach is the following:

1. Large-scale computation of percentiles. This uses the percentile range provided by the user as well as the minimum voxel size for the dataset you're using. This is a cellpose parameter that discards all regions of interests below this value. See [cellpose docs](https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py#L347).
2. Prediction of gradients in each axis, YX, ZX and ZY. We create a zarr dataset where we will output the predictions of each axis. It is important to mention that we have to normalize the data before processing, we have a local normalization option that assumes that step 1 was not computed and therefore only uses the data from that plane to compute the normalization. If avoided, this could result in segmentation masks merges between distinct cells and missing cells in dimmer areas. The global normalization step uses the precomputed percentiles from step 1.
3. Local combination of gradients, no overlapping areas are needed to perform this step. We create another zarr dataset that stores the combination of these gradients.
4. Following ZYX flows, computing histograms and generating cell centroids (seeds) in overlaped chunks. The ZYX flows and histograms are zarr datasets as well, however, these are computed in overlapping chunks and this overlap area must be $$overlapArea = 2*MeanCellDiameter$$ to avoid having cut-offs between the flows and affect the final segmentation mask.
5. Generation of segmentation masks in overlapped chunks. The overlapped chunks must have the exact same chunk area as step 4. We take all of the global seeds and assign individual cell IDs which then are assigned to the ZYX flow area.

## Disadvantages of this method.
1. We need enough disk storage to generate the final segmentation mask since we need to generate the gradients in each axis, combined gradients, ZYX flows and histograms. We recommend using a downsampled version of your data if your microscope has a very high resolution. It is only necessary to store the `segmentation_mask.zarr` and the other zarrs could be deleted without any problem.
2. Each chunk must have a size of: 
$$chunksize = (area_z, area_y, area_x) + (overlap_z * 2, overlap_y * 2, overlap_x * 2)$$ 
where the overlap is on each side of each axis.