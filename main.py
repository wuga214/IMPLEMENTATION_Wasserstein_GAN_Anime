import model.codings as nets
import tensorflow as tf

from utils.plot import show_curve, show_animes
from inputs.anime import Anime
from tqdm import tqdm
from model.wgan import GAN


def main():
    flags = tf.flags

    # VAE params
    flags.DEFINE_integer("latent_dim", 50, "Dimension of latent space.")
    flags.DEFINE_integer("observation_dim", 4096, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    # architectures
    flags.DEFINE_string("discriminator_architecture", 'conv', "Architecture to use for encoder.")
    flags.DEFINE_string("generator_architecture", 'conv', "Architecture to use for decoder.")

    # training params
    flags.DEFINE_integer("epochs", 20000,
                         "Total number of epochs for which to train the model.")
    flags.DEFINE_integer("updates_per_epoch", 1,
                         "Number of (mini batch) updates performed per epoch.")
    FLAGS = flags.FLAGS

    architectures = {
        'discriminator': {
            'conv': nets.gan_conv_anime_discriminator
        },
        'generator': {
            'conv': nets.gan_conv_anime_generator
        }
    }

    # define model
    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'observation_dim': FLAGS.observation_dim*3,
        'batch_size': FLAGS.batch_size,
        'generator': architectures['generator'][FLAGS.generator_architecture],
        'discriminator': architectures['discriminator'][FLAGS.discriminator_architecture]
    }
    gan = GAN(**kwargs)

    provider = Anime('faces')

    g_loss_curve = []
    d_loss_curve = []

    # do training
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        d_loss_cumm = 0.
        g_loss_cumm = 0.

        # iterate through batches
        for _ in range(FLAGS.updates_per_epoch):
            x = provider.next_batch(FLAGS.batch_size)
            d_loss, g_loss = gan.update(x, epoch)
            g_loss_cumm += g_loss
            d_loss_cumm += d_loss

        # average loss over most recent epoch
        g_loss_cumm /= (FLAGS.updates_per_epoch)
        d_loss_cumm /= (FLAGS.updates_per_epoch)

        g_loss_curve.append(g_loss_cumm)
        d_loss_curve.append(d_loss_cumm)
        # update progress bar
        s = "DLoss: {:.4f} Gloss: {:.4f}".format(d_loss_cumm, g_loss_cumm)
        tbar.set_description(s)

        # make pretty pictures if latent dim. is 2-dimensional
        if epoch % 100 == 0:
            show_animes(gan, FLAGS.batch_size, FLAGS.latent_dim, epoch)

    print "Train Done!"


    show_curve(range(FLAGS.epochs), g_loss_curve, 'Generator', True)
    show_curve(range(FLAGS.epochs), d_loss_curve, 'Discriminator', True)

    print "Curve Saving Done!"

    gan.save_generator('weights/gan_anime/gan_generator.tensorflow')


if __name__ == '__main__':
    main()