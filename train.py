# with help from https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models.simple_gan import Generator, ResidualGenerator, Discriminator
from utils.colour_transforms.hsv import rgb2hsv_palettes, hsv2rgb_palettes

np.random.seed(1)
SAVE_RESULTS = False
HSV = True
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

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
image_palettes = np.load("image_palettes.npy")
random_palettes = np.load("random_palettes.npy")

#source_palettes = image_palettes
source_palettes = random_palettes

if HSV:
    pastel_palettes = rgb2hsv_palettes(pastel_palettes)
    source_palettes = rgb2hsv_palettes(source_palettes)

N, D = pastel_palettes.shape

# parameters
model_name = "test"  # name to save model under
epochs = 60000
batch_size=191
learning_rate=0.0001
layer_size=8
n_batches = N // batch_size

params = {
    'batch_size': batch_size,
    'shuffle': False,  # already shuffled
    'num_workers': 1
}
history = {
        'epoch': [],
        'gen_loss': [],
        'dis_loss': []
    }

# models
generator = ResidualGenerator(D, layer_size)
#generator = Generator(D, layer_size)
discriminator = Discriminator(D, layer_size)
generator = generator.to(device)
discriminator = discriminator.to(device)

# optimisers
gen_optimiser = torch.optim.Adam(generator.parameters(), lr=learning_rate)#, weight_decay=1e-4)
dis_optimiser = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)#, weight_decay=1e-4)

# loss
loss = nn.BCELoss()

#
# training
#

for epoch in range(epochs):
    total_gen_loss = 0
    total_dis_loss = 0
    np.random.shuffle(source_palettes)
    np.random.shuffle(pastel_palettes)
    for b in range(n_batches):
        #source_batch, target_batch = source_palettes[b * batch_size:(b + 1) * batch_size, :], \
        #                             pastel_palettes[b * batch_size:(b + 1) * batch_size]
        source_batch, target_batch = source_palettes[b * batch_size:(b + 1) * batch_size, :], \
                                         np.random.rand(batch_size, D)
        source_batch, target_batch = torch.from_numpy(source_batch), torch.from_numpy(target_batch)
        source_batch, target_batch = source_batch.to(device, dtype=torch.float), \
                                     target_batch.to(device, dtype=torch.float)
        true_labels, false_labels = torch.ones((batch_size, 1), device=device), \
                                    torch.zeros((batch_size, 1), device=device)

        # zero the gradients
        gen_optimiser.zero_grad()

        generated_batch = generator(source_batch)

        # train generator
        gen_dis_out = discriminator(generated_batch)
        gen_loss = loss(gen_dis_out, true_labels)  # i.e. the discriminator should mark all generated files as 'true'
        gen_loss.backward()
        gen_optimiser.step()

        # train discriminator on true/generated data
        dis_optimiser.zero_grad()
        true_dis_out = discriminator(target_batch)
        true_dis_loss = loss(true_dis_out, true_labels)

        gen_dis_out = discriminator(generated_batch.detach())
        gen_dis_loss = loss(gen_dis_out, false_labels)
        dis_loss = (true_dis_loss + gen_dis_loss) / 2
        dis_loss.backward()
        dis_optimiser.step()

        total_gen_loss += gen_loss.item()
        total_dis_loss += dis_loss.item()
    epoch_gen_loss = total_gen_loss / n_batches
    epoch_dis_loss = total_dis_loss / n_batches
    history['epoch'].append(epoch)
    history['gen_loss'].append(epoch_gen_loss)
    history['dis_loss'].append(epoch_dis_loss)
    print("epoch {}: generator loss = {:.5g}, discriminator loss = {:.5g}".format(epoch, epoch_gen_loss, epoch_dis_loss))

plt.plot(history['gen_loss'], label="generator loss")
plt.plot(history['dis_loss'], label="discriminator loss")
plt.legend()
plt.show()

#
# testing results
#
model_name = "randgen-{}_{}_{}_{}".format(epochs, batch_size, learning_rate, layer_size)

#source_palettes = image_palettes
num_samples = 5
with torch.no_grad():
  source_palette = source_palettes[:num_samples, :]
  #source_palette = np.random.rand(num_samples, D)
  source_palette = torch.from_numpy(source_palette)
  source_palette = source_palette.to(device, dtype=torch.float)
  gen_palette = generator(source_palette)

if HSV: source_palettes = hsv2rgb_palettes(source_palettes)
original_palette = [vector2im(source_palettes[i, :]) for i in range(num_samples)]
gen_palette = gen_palette.cpu().detach().numpy()
if HSV: gen_palette = hsv2rgb_palettes(gen_palette)
gen_palette = [vector2im(gen_palette[i, :]) for i in range(num_samples)]

for i in range(num_samples):
    plt.subplot(1, 2*num_samples, (2*i)+1)
    plt.imshow(original_palette[i])
    plt.title("original palette")
    plt.subplot(1, 2*num_samples, (2*i)+2)
    plt.imshow(gen_palette[i])
    if SAVE_RESULTS: plt.savefig("results/{}_demo_palettes.png".format(model_name))
    plt.title("generated palette")
plt.show()


if SAVE_RESULTS: torch.save(generator.state_dict(), "checkpoints/{}.pt".format(model_name))
