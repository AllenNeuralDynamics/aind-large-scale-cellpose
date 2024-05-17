FROM registry.codeocean.allenneuraldynamics.org/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-large-scale-prediction==1.0.0 \
    scikit-image==0.22.0 \
    matplotlib==3.9.0 \
    scipy==1.11.3 \
    cellpose==3.0.5 \
    numba==0.59.1