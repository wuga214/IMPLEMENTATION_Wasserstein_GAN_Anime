import matplotlib.pyplot as plt
import numpy as np


def show_curve(epochs, losses, loss_type='Generator', save=False):
    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(loss_type)
    if save:
        plt.tight_layout()
        plt.savefig('figs/loss/{0}.eps'.format(loss_type), format='eps')
    else:
        plt.show()


def show_animes(gan, batch_size, dim_size, epoch, n=10, save=True):
    """
    Creates and saves a 'canvas' of images decoded from a grid in the latent space.
    :param vae: instance of VAE which performs decoding
    :param batch_size: little hack to get dimensions right
    :param epoch: current epoch of training
    :param n: number of points in each dimension of grid
    :param bound: samples from [-bound, bound] in both z1 and z2 direction
    """
    # create grid (could be done once but requires refactoring)
    noise = np.random.normal(size=(batch_size, dim_size))
    images = gan.z2x(noise, MNIST=False)
    images = (images+1)/2

    # create and fill canvas
    canvas = np.empty((64 * n, 64 * n, 3))
    for i in range(n):
        for j in range(n):
            canvas[(n - i - 1) * 64:(n - i) * 64, j * 64:(j + 1) * 64, 0:3] = images[i*n+j]

    # make figure and save
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig('figs/canvas_anime/' + str(epoch + 1000) + '.pdf',format='pdf')
    else:
        plt.show()