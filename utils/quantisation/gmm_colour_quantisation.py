import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.transform import resize
from os.path import join

imsource = "../../images"

image_name = "kth_campus.jpg"

image_path = join(imsource, image_name)

#
# load image and prepare pixel data
#
print("loading and preparing image...")
target_scale = 1/8
image = plt.imread(image_path)
image = resize(image, (int(image.shape[0]*target_scale), int(image.shape[1]*target_scale), image.shape[2]))
print(image.shape)
print(image[0,0,:])
pixels = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
print(pixels[0])

#
# train Gaussian mixture model.
# We want to find 5 clusters to actually make the comparison with colormind at
# http://colormind.io/blog/extracting-colors-from-photos-and-video/, but for the pastel learning approach we'll need 4.
#
print("training gmm model...")
n_components = 5
gmm = GaussianMixture(n_components=n_components)
gmm.fit(pixels)
print(gmm.means_)
print(gmm.means_.shape)
print(gmm.means_[0])

#
# display resulting colours
#
print("displaying colours...")
colours = np.reshape(gmm.means_, (gmm.means_.shape[0], 1, gmm.means_.shape[1]))
print(colours[0, 0, :])

plt.imshow(colours)
plt.show()
