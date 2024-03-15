# aind-z1-cell-segmentation

Large-scale cell segmentation using cellpose for Z1 data.

The approach is the following:

1. Prediction of gradients in each axis.
2. Local combination of gradients.
3. Following flows and generating centers (seed) of each cell in overlaped chunks.
4. Generation of segmentation masks in overlaped chunks.