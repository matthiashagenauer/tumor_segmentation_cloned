FROM nvcr.io/nvidia/pytorch:22.02-py3

WORKDIR /workdir

COPY . .

# Install python packages
RUN python -m pip install --upgrade pip  && python -m pip uninstall opencv_python && python -m pip install --no-cache-dir \
    pandas==1.4.1 \
    opencv_python_headless==4.5.5.64 \
    albumentations==1.1.0 \
    timm==0.4.12 \
    segmentation_models_pytorch==0.2.1 \
    torchinfo==1.6.3 \
    toml==0.10.2
