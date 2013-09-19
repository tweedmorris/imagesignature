from __future__ import division

import numpy as np
from scipy.misc import imresize
import skimage.color as skcolor

from scipy.fftpack import dct, idct
from scipy.ndimage.filters import gaussian_filter


def signatureSal(img, color_channels='lab', blur_sigma=0.045, map_width=64, resize_to_input=True, subtract_min=True):
    """
    Compute saliency map based on the Image Signature as described in:
        "Image Signature: Highlighting sparse salient regions",
        Xiaodi Hou, Jonathan Harel, and Christof Koch,
        IEEE Trans. Pattern Anal. Mach. Intell. 34(1):194-201 (2012)

    Parameters:
        im: (ndarray)
            Three-channel RGB image or (single-channel) grayscale image
        color_channels: (string)
            Which channels of color image to use ('Lab' or 'RGB')
        blur_sigma: (float)
            Amount of blur to apply to signature output, as fraction of image width
        map_width: (int)
            Size of the underlying saliency map
        resize_to_input: (boolean)
            Whether to resize saliency map to input image size; can be slow
        subtract_min: (boolean)
            Whether to normalize lowest saliency pixels to have value 0

    Returns:
        Single-channel ndarray of same width/height proportion as input image

    """
    h,w = img.shape[:2]

    img = imresize(img / 255., map_width / w)
    num_channels = img.shape[2] if len(img.shape) > 2 else 1
    color_channels = color_channels.lower()


    if num_channels == 1:
        t_img = img[...,np.newaxis]
    elif num_channels == 3:
        if color_channels == 'rgb':
            if img.dtype == 'uint8':
                t_img = img / 255.
            else:
                t_img = img
        elif color_channels == 'lab':
            t_img = skcolor.rgb2lab(img)
        elif color_channels == 'dkl':
            raise ValueError("rgb-to-dkl conversion not supported")
        else:
            raise ValueError("unknown color channels")
    else:
        raise ValueError("expect single-channel or 3-channel image")

    c_sal_map = np.dstack([
        idct2(np.sign(dct2(t_img[...,i]))) ** 2 for i in range(num_channels)
    ])

    out_map = c_sal_map.mean(axis=2)

    if blur_sigma > 0:
        kSize = map_width * blur_sigma
        out_map = gaussian_filter(out_map, kSize)

    if resize_to_input:
        out_map = imresize(out_map, (h,w)) / 255.

    # Normalize output
    if subtract_min:
        out_map -= out_map.min()
    out_map /= out_map.max()

    return out_map


# Wrappers around the DCT type 2 and 3 functions supplied by FFT pack,
# to employ same normalization as used by Matlab
def dct2(im):
    return dct(dct(im.T, norm='ortho').T, norm='ortho')

def idct2(im):
    return idct(idct(im.T, norm='ortho').T, norm='ortho')


# Nice heatmap overlay function as seen in Matlab implementation
def heatmap_overlay(img, overlay, colormap='jet', opacity=0.8, gamma=0.8):
    if heatmap_overlay._colormaps.has_key(colormap):
        heatmap_colors = heatmap_overlay._colormaps[colormap]
    else:
        heatmap_colors = cm.jet(range(256))[...,:3]
        heatmap_overlay._colormaps[colormap] = heatmap_colors

    overlay = overlay ** gamma
    overlay_3d = overlay[...,np.newaxis]

    if img.dtype == 'uint8':
        img = img / 255.

    return np.clip(
        opacity * (1 - overlay_3d) * img + overlay_3d * heatmap_colors[(overlay * 255).astype('uint8')],
        0, 1
    )

heatmap_overlay._colormaps = {}


if __name__ == '__main__':
    # Look for example pictures in the samplepics/ directory,
    # show the saliency signature for each.

    import glob
    from scipy.misc import imread
    from matplotlib import pyplot, cm

    for img_num, img_name in enumerate(glob.glob('samplepics/*.jpg'), 1):
        print('Computing saliency maps for sample image {0} ...'.format(img_num))

        img = imread(img_name)
        lab_map = signatureSal(img)
        rgb_map = signatureSal(img, color_channels='rgb')

        pyplot.subplot(121).title.set_text('Sample Image {0}: Image Signature - LAB'.format(img_num))
        pyplot.imshow(heatmap_overlay(img, lab_map))
        pyplot.subplot(122).title.set_text('Sample Image {0}: Image Signature - RGB'.format(img_num))
        pyplot.imshow(heatmap_overlay(img, rgb_map))
        pyplot.show()
