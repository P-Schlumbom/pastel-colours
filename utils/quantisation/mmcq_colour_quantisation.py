import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from os.path import join

from utils.quantisation.mmcq import mmcq

VERBOSE = 1

imsource = "../../images"
image_name = "kth_campus.jpg"
image_path = join(imsource, image_name)
target_scale = 1/1
n_colours = 8

#
# load image and prepare pixel data
#
if VERBOSE >= 1: print("loading and preparing image...")
image = plt.imread(image_path)
image = resize(image, (int(image.shape[0]*target_scale), int(image.shape[1]*target_scale), image.shape[2]))
if VERBOSE >= 2:
    print(image.shape)
    print(image[0,0,:])

#
# Use MMCQ to produce colour scheme.
# Note that due to how MMCQ works, the number of components should be 2^x for some integer x.
#
if VERBOSE >= 1: print("running MMCQ algorithm...")
colours = mmcq(image, n_colours)

#
# display resulting colours
#
if VERBOSE >= 1: print("displaying colours...")
colours = np.asarray(colours)
colours = np.reshape(colours, (colours.shape[0], 1, colours.shape[1]))
if VERBOSE >= 2: print(colours[0, 0, :])

plt.imshow(colours)
plt.show()
