FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U "pip<26" setuptools wheel

# 1) Install Waymo deps except jax/jaxlib
RUN pip install --no-cache-dir \
    absl-py==1.4.0 \
    dask[dataframe]==2023.3.1 \
    einsum==0.3.0 \
    google-auth==2.16.2 \
    immutabledict==2.2.0

# 2) Install jax/jaxlib from JAX wheel index
RUN pip install --no-cache-dir \
    "jax==0.4.13" \
    "jaxlib==0.4.13" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html


RUN pip install --no-cache-dir "numpy==1.24.3"
# TF (CPU build)
RUN pip install --no-cache-dir "tensorflow-cpu==2.12.*"

# Decision Transformer training dependencies.
RUN pip install --no-cache-dir \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    "transformers>=4.46,<5.0" \
    "google-cloud-storage"

# Notebook stack compatible with TF 2.12 pins
RUN pip install --no-cache-dir \
  "exceptiongroup<1.2" \
  "jupyterlab<4.2" \
  "ipykernel<7" \
  "ipython<8.13" \
  "matplotlib<3.9" \
  "tqdm"

# 3) Install Waymo wheel
RUN pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7


WORKDIR /work
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]

RUN python -c "import tensorflow as tf; from importlib.metadata import version; \
print('tf', tf.__version__); \
print('typing_extensions', version('typing_extensions')); \
print('exceptiongroup', version('exceptiongroup'))"
