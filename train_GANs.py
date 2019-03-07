from GANImageRegistration import GANImageRegistration
import numpy as np


def generate_data(crop_sz=(64,64,64) ):
    pass



def train(self, epochs, batch_size=128, sample_interval=50):

    imgRegGan = GANImageRegistration()
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise and generate img
        z = np.random.normal(size=(batch_size, self.latent_dim))
        phi = imgRegGan.generator.predict(z)

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        z_ = imgRegGan.transformation.predict([imgs, phi])

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss_real = imgRegGan.discriminator.train_on_batch([z_, imgs], valid)
        d_loss_fake = imgRegGan.discriminator.train_on_batch([z, imgs_], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (z -> img is valid and img -> z is is invalid)
        g_loss = imgRegGan.generator.train_on_batch([z, imgs], [valid, fake])

        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

        # # If at save interval => save generated image samples
        # if epoch % sample_interval == 0:
        #     self.sample_interval(epoch)


