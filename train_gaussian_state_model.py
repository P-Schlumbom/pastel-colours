from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models.gaussian_state_model import GaussianStateMachine
from utils.colour_transforms.hsv import rgb2hsv_palettes, hsv2rgb_palettes

np.random.seed(1)
SAVE_RESULTS = False
HSV = True
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

modelpath = "checkpoints/GSM"

#
# helper functions
#

def vector2im(vector):
    return np.reshape(vector, (4, 1, 3))

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
modelname = "test"
gsm = GaussianStateMachine(4, 3)  # 4 states, with 3 Gaussian components for each state
gsm.train_model(pastel_palettes_set)
gsm.save_model(join(modelpath, modelname + ".pkl"))
#print("done")
#gsm.load_model(join(modelpath, modelname + ".pkl"))

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


plt.subplot(1, 2, 1)
plt.imshow(original_palette)
plt.title("original palette")
plt.subplot(1, 2, 2)
plt.imshow(gen_palette)
plt.title("generated palette")
if SAVE_RESULTS: plt.savefig("results/gsm_out/{}_demo_palettes.png".format(modelname))
plt.show()

