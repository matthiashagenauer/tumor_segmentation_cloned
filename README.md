# Automatic tumour segmentation in WSIs

Segmentation of whole-slide images (WSIs) of H&E stained histological slides into
foreground (cancerous tissue) and background (everything else).

This repository contains code used to develop and validate the method presented in the
study titled *Generalisation of automatic tumour segmentation in histopathological
whole-slide images across multiple cancer types* ([https://arxiv.org/abs/2510.11182](https://arxiv.org/abs/2510.11182))

## Docker

In the project root directory, build the image defined by the `Dockerfile`.
Change the `-t <IMAGE_NAME>:<TAG>` as you like. E.g.

```
$ docker build -t name/tumour-segmentation:v01 .
```

Build an interactive container from this image.

```
$ docker run -it --gpus all --ipc host -v <SRC_PATH_1>:<DEST_PATH_1> -v <SRC_PATH_2>:<DEST_PATH_2> --name <CONTAINER_NAME> <IMAGE_NAME>:<TAG> bash
```

Once inside, you should be able to run all programs in this project.

Software dependencies with version information is provided in the supplied `Dockerfile`.

## Segmentation training

Given an input list of tiles with associated masks, one can train a model from scratch.
Code for network training and model inference is located at `process`.
There is also a `Dockerfile` specific for this purpose which is based on the NVCR
pytorch image so training should be faster.
All unneeded packages are removed compared to the root `Dockerfile`, but the included
python packages should be the same version.

Example command with distributed training using torch distributed data parallel with 1
node and 8 GPUs per node:

```
cd process
torchrun --nproc_per_node=8 src/run_training.py --config config/train.toml --input /path/to/input.csv --output /path/to/output
```

The input configuration file is the same as was used to create the published model.
The input .csv is a table with image tiles and associated masks, and is required to be on the
form

```
ImagePath,MaskPath
/path/to/image.extension,/path/to/corresponding_mask.extension
...
```

The default training configuration is set up with a DGX-A100 in mind (8 GPUs, where each GPU
has minimum 40 GB memory). Memory requirements are determined by the input size (batch
size and target tile size) and neural network architecture (encoder and decoder).

## Segmentation inference

All parts required for segmenting a WSI (preprocessing, neural network inference, and
postprocessing) are provided as runnable applications, and are combined in a script
called `full_scan_segmentation.py` for convenience.

This program excepts both a single scan, or a collection of scans either given as an
input `.csv` file or as multiple input paths.

Example:

```
$ python full_scan_segmentation.py /path/to/scans/*.svs /path/to/output /path/to/local-cache /path/to/model.tar
```

Make sure that the tile merging program (which is written in rust) is compiled. To do
so, enter `preprocess/tile_with_overlap` and run

```
cargo build --release
```

Inference with the tile size used for the published model (7680 x 7680) requires a GPU
with minimum 24 GB memory.
