# Setup cell.
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gan_pytorch import preprocess_img, deprocess_img, rel_error, count_params, ChunkSampler
from gan_pytorch import Flatten, Unflatten, initialize_weights
from gan_pytorch import discriminator, build_dc_classifier
from gan_pytorch import generator, build_dc_generator
from gan_pytorch import bce_loss, discriminator_loss, generator_loss
from gan_pytorch import ls_discriminator_loss, ls_generator_loss
from gan_pytorch import get_optimizer, run_a_gan
from gan_pytorch import sample_noise

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # Set default size of plots.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_images(images, epoch_cnt=0):
    images = np.reshape(images, [images.shape[0], -1])  # Images reshape to (batch_size, D).
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    plt.savefig(f"./out/gen_iter_{(epoch_cnt + 1) * 250}.png")
    return


answers = dict(np.load('gan-checks.npz'))
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

mnist_train = dset.MNIST(
    './datasets/MNIST_data',
    train=True,
    download=True,
    transform=T.ToTensor()
)
loader_train = DataLoader(
    mnist_train,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_TRAIN, 0)
)

mnist_val = dset.MNIST(
    './datasets/MNIST_data',
    train=True,
    download=True,
    transform=T.ToTensor()
)
loader_val = DataLoader(
    mnist_val,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
)

imgs = loader_train.__iter__().__next__()[0].view(batch_size, 784).numpy().squeeze()


def go_gan():
    # Make the discriminator
    D = discriminator().type(dtype)

    # Make the generator
    G = generator().type(dtype)

    # Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)

    # Run it!
    images = run_a_gan(
        D,
        G,
        D_solver,
        G_solver,
        ls_discriminator_loss,
        ls_generator_loss,
        loader_train,
        num_epochs=20
    )

    for i in range(len(images)):
        show_images(images[i], i)

def go_lsgan():
    # Make the discriminator
    D = discriminator().type(dtype)

    # Make the generator
    G = generator().type(dtype)

    # Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)

    # Run it!

    images = run_a_gan(
        D,
        G,
        D_solver,
        G_solver,
        ls_discriminator_loss,
        ls_generator_loss,
        loader_train,
        num_epochs=20
    )

    for i in range(len(images)):
        show_images(images[i], i)

def go_dcgan():

    D = build_dc_classifier().to(device)
    D.apply(initialize_weights)

    G = build_dc_generator().to(device)
    G.apply(initialize_weights)

    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)

    images = run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,
              loader_train, num_epochs=20)

    for i in range(len(images)):
        show_images(images[i], i)

go_dcgan()