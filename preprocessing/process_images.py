# given a directory of images, calculate the colour scheme for each one and store it

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from os import walk, listdir
from os.path import join

from utils.quantisation.mmcq import mmcq


datapath = "C:/Users/GGPC/Documents/Datasets/image_collection"
savepath = ".."
n_colours = 4
target_scale = 1 / 8

N = len(listdir(datapath))
image_palettes = np.zeros((N, 3*n_colours))

for i, im_name in enumerate(listdir(datapath)):
    print("\rProcessing image {}/{}: {}".format(i, N, im_name), end="")
    image_path = join(datapath, im_name)
    image = plt.imread(image_path)
    image = resize(image, (int(image.shape[0] * target_scale), int(image.shape[1] * target_scale), image.shape[2]))
    palette = mmcq(image, n_colours)
    image_palettes[i, :] = np.concatenate(palette)[:3*n_colours]  # in case image had 4 channels

    if i % 25 == 0:
        colours = np.asarray(palette)
        colours = np.reshape(colours, (colours.shape[0], 1, colours.shape[1]))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(colours)
        plt.show()

np.save(join(savepath, "image_palettes.npy"), image_palettes)

