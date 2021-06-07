# pretty straightforward explanation here: https://en.wikipedia.org/wiki/Median_cut
# otherwise also look here: https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
# here: http://leptonica.org/papers/mediancut.pdf
# or here: https://github.com/kanghyojun/mmcq.py/blob/master/mmcq/quantize.py

import numpy as np

def mmcq_loop(pixels, q):
    """
    The main recursive loop of the MMCQ algorithm. Returns list of quantised pixels from each bin or itself.
    :param pixels: numpy array of shape (N, 3), of N RGB pixels
    :param q: int, quantisation level. Recursion stops when this reaches zero.
    :return: a list of numpy arrays of shape (1, 3), where each array represents the mean pixel value of a final leaf
    bin.
    """
    variances = np.std(pixels, axis=0)
    sort_col = np.argmax(variances)
    pixels = pixels[pixels[:, sort_col].argsort()]
    halfway_index = pixels.shape[0] // 2
    if q > 0 and halfway_index > 0:
        return mmcq_loop(pixels[:halfway_index, :], q - 1) + mmcq_loop(pixels[halfway_index:, :], q - 1)
    return [np.mean(pixels, axis=0)]

def mmcq(im, n_bins=2):
    q = int(np.log2(n_bins))
    pixels = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    quantised_pixels = mmcq_loop(pixels, q)
    return quantised_pixels

