# with help from https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models.simple_gan import Generator, Discriminator

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

#
# training set up
#
# prepare data
pastel_palettes = np.load("pastel_palettes.npy") / 255.
image_palettes = np.load("image_palettes.npy")
random_palettes = np.load("random_palettes.npy")

source_palettes = image_palettes

N, D = pastel_palettes.shape

# parameters
model_name = "test"  # name to save model under
epochs = 100
batch_size=16
learning_rate=0.0001
layer_size=64
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
generator = Generator(D, layer_size)
discriminator = Discriminator(D, layer_size)
generator = generator.to(device)
discriminator = discriminator.to(device)

# optimisers
gen_optimiser = torch.optim.Adam(generator.parameters(), lr=learning_rate)
dis_optimiser = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# loss
loss = nn.BCELoss()

#
# training
#

for epoch in range(epochs):
    total_gen_loss = 0
    total_dis_loss = 0
    for b in range(n_batches):
        source_batch, target_batch = source_palettes[b * batch_size:(b + 1) * batch_size, :], \
                                     pastel_palettes[b * batch_size:(b + 1) * batch_size]
        source_batch, target_batch = torch.from_numpy(source_batch), torch.from_numpy(target_batch)
        source_batch, target_batch = source_batch.to(device, dtype=torch.float), target_batch.to(device)
        true_labels, false_labels = torch.ones(batch_size), torch.zeros(batch_size)

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

