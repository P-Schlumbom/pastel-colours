from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models.gaussian_state_model import GaussianStateMachine
from utils.colour_transforms.hsv import rgb2hsv_palettes, hsv2rgb_palettes

np.random.seed(1)

SAVE_RESULTS = False  # if true, saves generated figures
LOAD_MODEL = True  # if true, loads previously trained model parameters. In this case modelpath and modelname must be
# set appropriately.
HSV = True  # if true, convert colours into HSV space. This makes an important difference, as pastel colours are much
# more clearly defined in HSV space.
modelpath = "checkpoints/GSM"  # path to the trained model files
modelname = "test"  # name of the model file to load. Do not include extension.

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True


#
# helper functions
#

def vector2im(vector):
    """
    Given a vector, convert it to a shape that can be read as an RGB image.
    :param vector: numpy array of shape 1xN where N = 3M and M is the number of colours
    :return: numpy array of shape (N//3)x1x3, i.e. an RGB image of dimensions (N//3)x1
    """
    n_colours = np.prod(vector.shape) // 3
    return np.reshape(vector, (n_colours, 1, 3))

#
# training set up
#
# prepare data
pastel_palettes = np.load("pastel_palettes.npy") / 255.

if HSV:
    pastel_palettes = rgb2hsv_palettes(pastel_palettes)

N, D = pastel_palettes.shape

pastel_palettes_set = [pastel_palettes[:,(i*3):(i*3)+3] for i in range(D//3)]
print(len(pastel_palettes_set))
print([pal.shape for pal in pastel_palettes_set])

#
# model setup
#

print("training model...")
gsm = GaussianStateMachine(4, 3)  # 4 states, with 3 Gaussian components for each state

#print("done")
if LOAD_MODEL:
    gsm.load_model(join(modelpath, modelname + ".pkl"))
else:
    gsm.train_model(pastel_palettes_set)
    gsm.save_model(join(modelpath, modelname + ".pkl"))

#
# evaluate model
#

sample_id = 0
print(pastel_palettes.shape)
source_palette = pastel_palettes[0, :]
gen_palette = gsm.sample(set_values={0:source_palette[:3].reshape(1, -1)})  # first colour only. reshaped for dimension.
source_palette = source_palette.reshape(1, -1)

if HSV: source_palette = hsv2rgb_palettes(source_palette)
original_palette = vector2im(source_palette)
#gen_palette = gen_palette.cpu().detach().numpy()
gen_palette = gen_palette.reshape(1, -1)
if HSV: gen_palette = hsv2rgb_palettes(gen_palette)
gen_palette = vector2im(gen_palette)

def generate_samples(source_palette, known_ids):
    """
    Given a source palette and a mask, use the model to generate a new palette conditioned on those colours in the
    original palette revealed by the mask.
    :param source_palette: numpy array of shape 1xN, representing the M RGB vectors of the palette's colours
    concatenated into a single vector. That is, N = 3 x M. The first colour is the palette's primary colour, the second
    the secondary, etc.
    :param known_ids: list of integers, where each integer indexes a colour in the original palette which is known,
    and which will be used by the generator to condition the generation of the remaining unknown colours.
    :return: 3 numpy arrays of shape Mx1x3 (i.e. a 3-channel image of shape Mx1) corresponding to the original, masked
    (all unknown colours in the palette set to black), and generated colour palettes, respectively.
    """
    set_values = {i:source_palette[(i*3):(i*3)+3].reshape(1, -1) for i in known_ids}
    gen_palette = gsm.sample(set_values=set_values)

    source_palette = source_palette.reshape(1, -1)
    if HSV: source_palette = hsv2rgb_palettes(source_palette)
    gen_palette = gen_palette.reshape(1, -1)
    if HSV: gen_palette = hsv2rgb_palettes(gen_palette)

    # mask original palette
    mask = np.zeros(source_palette.shape)
    for i in known_ids:
        mask[0, (i*3):(i*3)+3] = 1
    masked_palette = source_palette * mask
    #masked_palette = source_palette

    original_palette = vector2im(source_palette)
    masked_palette = vector2im(masked_palette)
    gen_palette = vector2im(gen_palette)
    return original_palette, masked_palette, gen_palette

def display_generations(source_palette):
    """
    Given a single source palette, display the generated palettes for 12 different masks applied to the source palette.
    :param source_palette: numpy array of shape 1xN, representing the M RGB vectors of the palette's colours
    concatenated into a single vector. That is, N = 3 x M. The first colour is the palette's primary colour, the second
    the secondary, etc.
    :return:
    """
    masks = [[0], [1], [2], [3],
             [0, 1], [1, 2], [2, 3], [0, 2],
             [1, 3], [0, 3], [0, 1, 2], [1, 2, 3]]
    rows, columns = 3, 4*2
    for i, m in enumerate(masks):
        original_palette, masked_palette, gen_palette = generate_samples(source_palette, m)
        plt.subplot(rows, columns, (i * 2) + 1)
        plt.imshow(masked_palette)
        plt.title("masked palette")
        plt.subplot(rows, columns, (i * 2) + 2)
        plt.imshow(gen_palette)
        plt.title("generated palette")
    if SAVE_RESULTS: plt.savefig("results/gsm_out/{}_palette_masks.png".format(modelname))
    plt.show()

def display_samples():
    """
    Display the generated palettes for each of the first 12 pastel palettes in the dataset. For each case, all colours
    but the first are masked, i.e. the palettes are generated based on the primary colour only.
    :return:
    """
    mask = [0]
    rows, columns = 3, 4*3
    for i, source_palette in enumerate(pastel_palettes[:12,:]):
        original_palette, masked_palette, gen_palette = generate_samples(source_palette, mask)
        plt.subplot(rows, columns, (i * 3) + 1)
        plt.imshow(original_palette)
        plt.title("original")
        plt.subplot(rows, columns, (i * 3) + 2)
        plt.imshow(masked_palette)
        plt.title("masked")
        plt.subplot(rows, columns, (i * 3) + 3)
        plt.imshow(gen_palette)
        plt.title("generated")
    if SAVE_RESULTS: plt.savefig("results/gsm_out/{}_palette_samples.png".format(modelname))
    plt.show()

def display_sample(source_palette):
    """
    Given a single source palette, display the palette, its masked version, and the corresponding generated palette.
    :param source_palette: numpy array of shape 1xN, representing the M RGB vectors of the palette's colours
    concatenated into a single vector. That is, N = 3 x M. The first colour is the palette's primary colour, the second
    the secondary, etc.
    :return:
    """
    original_palette, masked_palette, gen_palette = generate_samples(source_palette, [0])
    plt.subplot(1, 3, 1)
    plt.imshow(original_palette)
    plt.title("original")
    plt.subplot(1, 3, 2)
    plt.imshow(masked_palette)
    plt.title("masked")
    plt.subplot(1, 3, 3)
    plt.imshow(gen_palette)
    plt.title("generated")

    if SAVE_RESULTS: plt.savefig("results/gsm_out/{}_palette_samples.png".format(modelname))
    plt.show()

"""plt.subplot(1, 2, 1)
plt.imshow(original_palette)
plt.title("original palette")
plt.subplot(1, 2, 2)
plt.imshow(gen_palette)
plt.title("generated palette")
if SAVE_RESULTS: plt.savefig("results/gsm_out/{}_demo_palettes.png".format(modelname))
plt.show()"""

sample_id = 1
#display_generations(pastel_palettes[sample_id, :])
display_samples()
display_sample(pastel_palettes[sample_id, :])

