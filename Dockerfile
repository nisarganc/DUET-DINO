# syntax=docker/dockerfile:1
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM continuumio/miniconda3:latest

ARG CONDA_ENV=.duet-dino
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip -y \
    && conda clean -afy

ENV CONDA_DEFAULT_ENV=${CONDA_ENV}
ENV PATH=/opt/conda/envs/${CONDA_ENV}/bin:/opt/conda/bin:${PATH}

WORKDIR /workspace/duet-dino

COPY . .

# Install requirements if your project has requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}" \
    >> /root/.bashrc

CMD ["/bin/bash"]