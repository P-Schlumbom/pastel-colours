import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

def rgb2hsv_vec(rgb_vec):
    """
    Given an RGB vector, convert it into an HSV vector
    :param rgb_vec: numpy vector of shape (3,), 3 RGB values
    :return: numpy array of shape (3,), the corresponding HSV vector
    """
    rgb_im = np.reshape(rgb_vec, (1, 1, 3), order='A')
    hsv_im = rgb2hsv(rgb_im)
    return hsv_im[0,0,:]

def hsv2rgb_vec(hsv_vec):
    """
    Given an RGB vector, convert it into an HSV vector
    :param hsv_vec: numpy vector of shape (3,), 3 RGB values
    :return: numpy array of shape (3,), the corresponding HSV vector
    """
    hsv_im = np.reshape(hsv_vec, (1, 1, 3), order='A')
    rgb_im = hsv2rgb(hsv_im)
    return rgb_im[0,0,:]

def rgb2hsv_palettes(rgb_palette, n_colours=4):
    """
    Convert array of RGB palettes into array of equivalent HSV palettes
    :param rgb_palette: numpy array of shape (N, D), where N is the number of samples (i.e. palettes) and D=3*n_colours,
    where n_colours is the number of colours
    :param n_colours: int, the number of colours represented in one palette vector. 4 by default.
    :return: numpy array of shape (N, D), the palettes with all RGB values converted into HSV values.
    """
    rgb_palette = np.asarray([rgb_palette[:, i*3:(i+1)*3] for i in range(n_colours)])
    hsv_palette = rgb2hsv(rgb_palette)
    hsv_palette = np.concatenate([hsv_palette[i] for i in range(n_colours)], axis=1)
    return hsv_palette

def hsv2rgb_palettes(hsv_palette, n_colours=4):
    """
    Convert array of HSV palettes into an array of equivalent RGB palettes
    :param hsv_palette: numpy array of shape (N, D), where N is the number of samples (i.e. palettes) and D=3*n_colours,
    where n_colours is the number of colours
    :param n_colours: int, the number of colours represented in one palette vector. 4 by default.
    :return: numpy array of shape (N, D), the palettes with all HSV values converted into RGB values.
    """
    hsv_palette = np.asarray([hsv_palette[:, i*3:(i+1)*3] for i in range(n_colours)])
    rgb_palette = hsv2rgb(hsv_palette)
    rgb_palette = np.concatenate([rgb_palette[i] for i in range(n_colours)], axis=1)
    return rgb_palette

if __name__=="__main__":
    test_rgb = np.asarray([1., 0., 0.])
    print(test_rgb)
    test_hsv = rgb2hsv_vec(test_rgb)
    print(test_hsv)
    ret_rgb = hsv2rgb_vec(test_hsv)
    print(ret_rgb)

    rgb_palettes = np.load("../../pastel_palettes.npy") / 255.
    print(rgb_palettes[0,:])
    hsv_palettes = rgb2hsv_palettes(rgb_palettes)
    print(hsv_palettes[0,:])
    rgb_palettes = hsv2rgb_palettes(hsv_palettes)
    print(rgb_palettes[0,:])

