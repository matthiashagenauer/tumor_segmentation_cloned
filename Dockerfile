FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /workdir

COPY . .

# Install dependencies
RUN apt-get update && apt-get install --yes \
    build-essential \
    git \
    curl \
    rsync \
    software-properties-common \
    openslide-tools

# Install Rust and append its environment to PATH
# Cargo is installed to $HOME/.cargo by default, which in this case is /root/.cargo.
# With this, 'cargo' and other commands are not available if a container is created as
# non-root with the flag '-u $(id -u):$(id -g)', even if /root/.cargo/bin is in $PATH.
# To avoid this, install cargo in the same directory as nvidia and cuda, namely
# '/usr/local'. The same goes for rustup.
ENV CARGO_HOME="/usr/local/.cargo" RUSTUP_HOME="/usr/local/.rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/usr/local/.cargo/bin:${PATH}"
RUN cargo build --manifest-path=preprocess/tile_with_overlap/Cargo.toml --release

# Install python packages
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir \
    pandas==1.4.1 \
    opencv_python_headless==4.5.5.64 \
    matplotlib==3.5.1 \
    albumentations==1.1.0 \
    timm==0.4.12 \
    segmentation_models_pytorch==0.2.1 \
    torchinfo==1.6.3 \
    toml==0.10.2 \
    openslide_python==1.1.2 \
    polars==0.20.4
