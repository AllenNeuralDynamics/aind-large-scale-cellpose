# hash:sha256:646907375b38a5250607ef152664c03ab7d944094b5af5794e7d1607464152a6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-large-scale-prediction==1.0.0 \
    scikit-image==0.22.0 \
    matplotlib==3.9.0 \
    scipy==1.11.3 \
    cellpose==3.0.5 \
    numba==0.59.1 \
    xarray_multiscale==2.1.0 \
    boto3==1.43.6