FROM registry.codeocean.allenneuraldynamics.org/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ARG new_pred_version="0.0.9"

RUN pip install -U --no-cache-dir \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git \
    scikit-image==0.22.0 \
    matplotlib \
    dask-image \
    scipy==1.11.3 \
    cellpose==3.0.5 \
    numba