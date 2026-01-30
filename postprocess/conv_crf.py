"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann


Cloned from original 28.01.2021:

Repo: https://github.com/MarvinTeichmann/ConvCRF
Commit: 09306378ebc76e38f91aeaf57be5ce36ec2b873c

Substantial modifications are made to the code, but not to the crf algorithms.
"""

import logging
import warnings
import math

import numpy as np
import torch


class ConvCRF:
    """
    Implements a generic CRF class.

    This class provides tools to build your own ConvCRF based model.
    """

    def __init__(self, num_pixels, num_classes, config, gpu):
        self.num_pixels = num_pixels
        self.num_classes = num_classes
        self.config = config
        self.kernel = None

        if not torch.cuda.is_available():
            logging.error("GPU requested but not avaible.")
            raise ValueError

        if type(num_pixels) is tuple or type(num_pixels) is list:
            self.height = num_pixels[0]
            self.width = num_pixels[1]
        else:
            self.num_pixels = num_pixels

        if config.convcomp:
            self.convcomp = torch.nn.Conv2d(
                num_classes, num_classes, kernel_size=1, stride=1, padding=0, bias=False
            ).to(torch.device(f"cuda:{gpu}"))
            conv2d_weight = torch.tensor(0.1 * math.sqrt(2.0 / num_classes))
            self.convcomp.weight.data.fill_(conv2d_weight)
        else:
            self.convcomp = None

    def clean_filters(self):
        self.kernel = None

    def add_pairwise_energies(self, feat_list, compat_list, norm_list):
        assert len(feat_list) == len(compat_list)

        self.kernel = MessagePassingCol(
            feat_list=feat_list,
            compat_list=compat_list,
            norm_list=norm_list,
            merge=self.config.merge,
            num_pixels=self.num_pixels,
            filter_size=self.config.filter_size,
            num_classes=self.num_classes,
            blur=self.config.blur,
            verbose=self.config.verbose,
        )

    def inference(self, unary, num_iter=5):
        lg_unary = torch.nn.functional.log_softmax(unary, dim=1, _stacklevel=5)
        prediction = lg_unary

        unary_weight = self.config.unary_weight
        message_weight = self.config.message_weight
        for i in range(num_iter):
            final_step = i == num_iter - 1
            message = self.kernel.compute(prediction)

            if self.config.convcomp:
                message = message + self.convcomp(message)

            prediction = unary_weight * lg_unary + message_weight * message

            if self.config.softmax and not final_step:
                prediction = torch.nn.functional.softmax(prediction, dim=1)

        if self.config.final_softmax:
            prediction = torch.nn.functional.softmax(prediction, dim=1)

        return prediction


class GaussCRF:
    """Implements ConvCRF with hand-crafted features.

    It uses the more generic ConvCRF class as basis and utilizes a config class to
    easily set hyperparameters and follows the design choices of:

    Philipp Kraehenbuehl and Vladlen Koltun,
    Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
    (arxiv.org/abs/1210.5644)
    """

    def __init__(self, config, shape, num_classes=None, gpu=0):
        self.config = config
        self.shape = shape
        self.num_classes = num_classes

        self.mesh = self._create_mesh().cuda()

        # Set parameters for appearance and smoothness kernels. Note that we are
        # defining the reciprocal as we are multiplying rather than dividing later whan
        # applying them.
        # Also note that we only use the first number in the list, that is, different
        # values for x or y position, or for r, g, b channels are not implemented yet
        smoothness_pos = 1.0 / config.smoothness.position[0]
        self.smoothness_param_pos = torch.tensor([smoothness_pos]).cuda()
        self.smoothness_compat = torch.tensor([config.smoothness.compat]).cuda()

        appearance_pos = 1.0 / config.appearance.position[0]
        self.appearance_param_pos = torch.tensor([appearance_pos]).cuda()
        appearance_col = 1.0 / config.appearance.color[0]
        self.appearance_param_col = torch.tensor([appearance_col]).cuda()
        self.appearance_compat = torch.tensor([config.appearance.compat]).cuda()

        self.CRF = ConvCRF(shape, num_classes, config, gpu)

        return

    def inference(self, unary, img, num_iter=5):
        """Run a forward pass through ConvCRF.

        Arguments:
        unary: torch.tensor with shape [bs, num_classes, height, width].
            The unary predictions. Logsoftmax is applied to the unaries during
            inference. When using CNNs don't apply softmax, use unnormalised output
            (logits) instead.

        img: torch.tensor with shape [bs, 3, height, width]
            The input image. Default config assumes image data in [0, 255].
        """
        bs, c, x, _ = img.shape

        smoothness_features = self.create_smoothness_features(
            self.smoothness_param_pos, bs
        )
        appearance_features = self.create_appearance_features(
            img,
            self.appearance_param_col,
            self.appearance_param_pos,
            bs,
        )

        smoothness_normalisation = self.config.smoothness.normalisation
        appearance_normalisation = self.config.appearance.normalisation
        if smoothness_normalisation not in ["no", "symmetric"]:
            raise NotImplementedError
        if appearance_normalisation not in ["no", "symmetric"]:
            raise NotImplementedError
        if self.config.merge:
            if smoothness_normalisation != "no":
                raise NotImplementedError
            if appearance_normalisation != "no":
                raise NotImplementedError

        self.CRF.add_pairwise_energies(
            [smoothness_features, appearance_features],
            [self.smoothness_compat, self.appearance_compat],
            [smoothness_normalisation, appearance_normalisation],
        )

        prediction = self.CRF.inference(unary, self.config.num_iter)

        self.CRF.clean_filters()
        return prediction

    def _create_mesh(self):
        hcord_range = [range(s) for s in self.shape]
        mesh = np.array(np.meshgrid(*hcord_range, indexing="ij"), dtype=np.float16)
        return torch.from_numpy(mesh)

    def create_smoothness_features(self, position_param, bs=1):
        return torch.stack(bs * [self.mesh * position_param])

    def create_appearance_features(self, img, colour_param, position_param, bs=1):
        norm_img = img * colour_param
        norm_mesh = self.create_smoothness_features(position_param, bs)
        return torch.cat([norm_mesh, norm_img], dim=1)


def show_memusage(name):
    device = torch.cuda.current_device()
    a = torch.cuda.memory_allocated(device) / 1024 / 1024
    r = torch.cuda.memory_reserved(device) / 1024 / 1024
    logging.info(f"{name:<20}: Allocated: {a:>8.0f} MiB. Reserved: {r:>8.0f} MiB")


def _get_ind(dz):
    if dz == 0:
        return 0, 0
    if dz < 0:
        return 0, -dz
    if dz > 0:
        return dz, 0


def _negative(dz):
    """
    Computes -dz for numpy indexing. Goal is to use as in array[i:-dz].

    However, if dz=0 this indexing does not work.
    None needs to be used instead.
    """
    if dz == 0:
        return None
    else:
        return -dz


class MessagePassingCol:
    """Perform the Message passing of ConvCRFs.

    The main magic happens here.
    """

    def __init__(
        self,
        feat_list,
        compat_list,
        norm_list,
        merge,
        num_pixels,
        num_classes,
        filter_size=5,
        blur=1,
        matmul=False,
        verbose=False,
        gpu=0,
    ):

        span = filter_size // 2
        assert filter_size % 2 == 1
        self.span = span
        self.filter_size = filter_size
        self.verbose = verbose
        self.blur = blur
        self.merge = merge
        self.num_pixels = num_pixels

        if not self.blur == 1 and self.blur % 2:
            raise NotImplementedError

        self.matmul = matmul

        self._gaus_list = []
        self._norm_list = []

        for features, compat, norm in zip(feat_list, compat_list, norm_list):
            gaussian = self._create_convolutional_filters(features)
            if norm == "no":
                self._norm_list.append(None)
            else:
                mynorm = self._get_normalisation(gaussian)
                self._norm_list.append(mynorm)

            gaussian = compat * gaussian
            self._gaus_list.append(gaussian)

    def _get_normalisation(self, gaus):
        # Save some memory by not saving intermediates
        # features = torch.ones([1, 1, self.num_pixels[0], self.num_pixels[1]]).cuda()
        # norm_out = self._compute_gaussian(features, gaussian=gaus)
        return 1.0 / torch.sqrt(
            self._compute_gaussian(
                torch.ones([1, 1, self.num_pixels[0], self.num_pixels[1]]).cuda(),
                gaussian=gaus,
            )
            + 1e-20
        )

    def _create_convolutional_filters(self, features):
        span = self.span
        bs = features.shape[0]

        if self.blur > 1:
            off_0 = (self.blur - self.num_pixels[0] % self.blur) % self.blur
            off_1 = (self.blur - self.num_pixels[1] % self.blur) % self.blur
            pad_0 = math.ceil(off_0 / 2)
            pad_1 = math.ceil(off_1 / 2)
            if self.blur == 2:
                assert pad_0 == self.num_pixels[0] % 2
                assert pad_1 == self.num_pixels[1] % 2

            features = torch.nn.functional.avg_pool2d(
                features,
                kernel_size=self.blur,
                padding=(pad_0, pad_1),
                count_include_pad=False,
            )

            num_pixels = [
                math.ceil(self.num_pixels[0] / self.blur),
                math.ceil(self.num_pixels[1] / self.blur),
            ]
            assert num_pixels[0] == features.shape[2]
            assert num_pixels[1] == features.shape[3]
        else:
            num_pixels = self.num_pixels

        gaussian = features.data.new(
            bs, self.filter_size, self.filter_size, num_pixels[0], num_pixels[1]
        ).fill_(0)

        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):

                dx1, dx2 = _get_ind(dx)
                dy1, dy2 = _get_ind(dy)
                ndx1 = _negative(dx1)
                ndx2 = _negative(dx2)
                ndy1 = _negative(dy1)
                ndy2 = _negative(dy2)

                feat_t = features[:, :, dx1:ndx2, dy1:ndy2]
                # NOQA
                feat_t2 = features[:, :, dx2:ndx1, dy2:ndy1]

                diff = feat_t - feat_t2
                diff_sq = diff * diff
                exp_diff = torch.exp(torch.sum(-0.5 * diff_sq, dim=1))

                gaussian[:, dx + span, dy + span, dx2:ndx1, dy2:ndy1] = exp_diff

        return gaussian.view(
            bs, 1, self.filter_size, self.filter_size, num_pixels[0], num_pixels[1]
        )

    def compute(self, prediction):
        if self.merge:
            gaussian = sum(self._gaus_list)
            result = self._compute_gaussian(prediction, gaussian)
        else:
            assert len(self._gaus_list) == len(self._norm_list)
            result = 0
            for gaus, normalisation in zip(self._gaus_list, self._norm_list):
                result += self._compute_gaussian(prediction, gaus, normalisation)

        return result

    def _compute_gaussian(self, in_tensor, gaussian, normalisation=None):
        if normalisation is not None:
            in_tensor = in_tensor * normalisation

        shape = in_tensor.shape
        num_channels = shape[1]
        bs = shape[0]

        if self.verbose:
            show_memusage("BeforeDownsampling")

        if self.blur > 1:
            off_0 = (self.blur - self.num_pixels[0] % self.blur) % self.blur
            off_1 = (self.blur - self.num_pixels[1] % self.blur) % self.blur
            pad_0 = int(math.ceil(off_0 / 2))
            pad_1 = int(math.ceil(off_1 / 2))
            in_tensor = torch.nn.functional.avg_pool2d(
                in_tensor,
                kernel_size=self.blur,
                padding=(pad_0, pad_1),
                count_include_pad=False,
            )
            num_pixels = [
                math.ceil(self.num_pixels[0] / self.blur),
                math.ceil(self.num_pixels[1] / self.blur),
            ]
            assert num_pixels[0] == in_tensor.shape[2]
            assert num_pixels[1] == in_tensor.shape[3]
        else:
            num_pixels = self.num_pixels

        if self.verbose:
            show_memusage("Init")

        # An alternative implementation of im2col.
        #
        # This has implementation uses the torch 0.4 im2col operation.
        # This implementation was not avaible when we did the experiments
        # published in our paper. So less "testing" has been done.
        #
        # It is around ~20% slower then the pyinn implementation but
        # easier to use as it removes a dependency.
        input_unfold = torch.nn.functional.unfold(
            in_tensor, self.filter_size, 1, self.span
        )
        input_col = input_unfold.view(
            bs,
            num_channels,
            self.filter_size,
            self.filter_size,
            num_pixels[0],
            num_pixels[1],
        )

        k_sqr = self.filter_size * self.filter_size

        if self.verbose:
            show_memusage("Im2Col")

        product = gaussian * input_col
        if self.verbose:
            show_memusage("Product")

        product = product.view([bs, num_channels, k_sqr, num_pixels[0], num_pixels[1]])
        message = product.sum(2)

        if self.verbose:
            show_memusage("FinalNorm")

        if self.blur > 1:
            in_0 = self.num_pixels[0]
            in_1 = self.num_pixels[1]
            message = message.view(bs, num_channels, num_pixels[0], num_pixels[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suppress warning regarding corner alignment
                message = torch.nn.functional.upsample(
                    message, scale_factor=self.blur, mode="bilinear"
                )

            message = message[:, :, pad_0 : pad_0 + in_0, pad_1 : in_1 + pad_1]
            message = message.contiguous()

            message = message.view(shape)
            assert message.shape == shape

        if normalisation is not None:
            message = normalisation * message

        if self.verbose:
            show_memusage("AfterUpsampling")

        return message


def run_segmentation(image, prob_fg, config, gpu):
    """
    prob_fg: 2d grayscale image with values in [0, 255]
    image: 3d rgb image with values in [0, 255]

    returns 2d bool image with values in {0, 1}
    """
    torch.set_default_dtype(torch.float16)
    torch.cuda.set_device(gpu)

    height = image.shape[0]
    width = image.shape[1]
    area = height * width

    # Adapt blur to save memory. Lower blur -> more memory is required. Larger images
    # allows for larger blur value without noticing the effects too much.
    if area > 1.5e8:
        assert False, f"Image is too large: {height} x {width} = {area}"
    elif area > 1.3e8:
        if config.verbose:
            logging.info("Area above 1.3e8 -> blur = 12")
        config.blur = 12
    elif area > 1e8:
        if config.verbose:
            logging.info("Area above 1.0e8 -> blur = 10")
        config.blur = 10
    elif area > 8e7:
        if config.verbose:
            logging.info("Area above 8.0e7 -> blur = 8")
        config.blur = 8
    elif area > 5e7:
        if config.verbose:
            logging.info("Area above 5.0e7 -> blur = 6")
        config.blur = 6
    else:
        if config.verbose:
            logging.info("Area below all -> blur = 4")
        config.blur = 4

    unary = (prob_fg / 255.0).astype(np.float16)
    unary = np.array([1.0 - unary, unary]).astype(np.float16)
    num_classes = unary.shape[0]

    # Define unary
    unary = (prob_fg / 255.0).astype(np.float16)
    unary = np.array([1.0 - unary, unary]).astype(np.float16)
    num_classes = unary.shape[0]

    # Make input pytorch compatible
    image = image.transpose(2, 0, 1)
    image = image.reshape([1, 3, height, width])
    image_var = torch.tensor(image)
    unary = unary.reshape([1, num_classes, height, width])
    unary_var = torch.tensor(unary)

    # Create CRF module
    gausscrf = GaussCRF(config, (height, width), num_classes, gpu)

    # Move to GPU
    image_var = image_var.cuda()
    unary_var = unary_var.cuda()

    # Perform CRF inference
    prediction = gausscrf.inference(unary=unary_var, img=image_var)

    prediction_np = prediction.data.cpu().numpy()
    prediction_np = prediction_np[0]
    mask = np.argmax(prediction_np, axis=0) > 0

    # This should not be necessary, but without it OOM Cuda errors are observed when
    # running multiple cases in sequence on one GPU and / or multiple cases in parallel
    # on multiple GPU.
    torch.cuda.empty_cache()

    return mask
