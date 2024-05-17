# aind-large-scale-cellpose

Large-scale cell segmentation using cellpose.

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

## Documentation
You can access the documentation for this module [here]().

## Contributing

To develop the code, install the packages described in the Dockerfile.

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 . --max-line-length=100
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
