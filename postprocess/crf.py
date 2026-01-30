import cv2
import numpy as np

import conv_crf
import utils


class KernelParams:
    def __init__(self, position, compat, normalisation, color=None):
        # 2D vector for x and y directions
        self.position = position
        # 2D vector for r, g, b channels
        self.color = color
        # {scalar, 1D list, 2D matrix}
        self.compat = compat
        # {"constant", "vector", "full"}. Currently unused
        self.kernel = "vector"
        # Values in {"no", "before", "after", "symmetric"}
        self.normalisation = normalisation


class Configurations:
    def __init__(self):

        self.verbose = False
        self.num_iter = 20
        self.target_factor = 0.2  # Resize with this factor ...
        self.min_area = 1.0e6  # unless result area is smaller than this

        self.smoothness = KernelParams([3, 3], 3, "symmetric")
        self.appearance = KernelParams([100, 100], 50, "symmetric", color=[5, 5, 5])

        self.unary_weight = 0.8
        self.message_weight = 0.2

        # Assume pixels i and j are conditionally independent (and therefore have a
        # pairwise potential equal to zero) if the distance between them is larger than
        # filter_size.
        self.filter_size = 11
        self.blur = 4
        # If normalisation is "no" setting this to True will speed up algorithm slightly
        self.merge = False
        self.convcomp = False
        self.softmax = True
        self.final_softmax = False


def prepare_input(image_bgr, probability_map, target_factor, min_area):
    height, width = probability_map.shape
    area = height * width
    if area > min_area:
        resized = True
        target_height, target_width = utils.resized_dim(
            height, width, target_factor, min_area
        )
        probability_map = cv2.resize(
            probability_map,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        image_bgr = cv2.resize(
            image_bgr,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = False

    probability_map = utils.sigmoid(probability_map / 255.0, 0.9) * 255.0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, probability_map, resized


def segment_probability(image_bgr, probability_map, gpu):
    """
    Apply CRF on the original image using the probability map as unary potential.

    The gpu version (conv crf) is about 50 times faster than the original (dense crf) on
    megapixel images. Both gives similar results but the convolutional version relies on
    an assumption that is not present in the original. This assumption allows the
    convolutional version to utilise convolutions in stead of the permutohedral lattice
    in message passing, otherwise they are equal.
    """
    config = Configurations()
    height, width = probability_map.shape
    image_rgb, probability_map, resized = prepare_input(
        image_bgr, probability_map, config.target_factor, config.min_area
    )
    mask = conv_crf.run_segmentation(image_rgb, probability_map, config, gpu)
    if resized:
        mask = cv2.resize(
            mask.astype(np.uint8) * 255,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        mask = mask > 127

    return mask
