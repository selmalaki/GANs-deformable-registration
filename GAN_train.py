
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, CSVLogger

from ImageRegistrationGANs.data_loader import DataLoader
from ImageRegistrationGANs.GAN_unet_model40 import GANUnetModel40

import os
import numpy as np



class GANUnetTrain:

    def __init__(self):

        self.batch_sz = 16
        self.network = GANUnetModel40()
        self.data_loader = DataLoader(batch_sz=self.batch_sz)


    def train_network(self, steps_per_epoch=100, epochs=2000, n_gpus=1, save_name=None, validation_data=None):
        if save_name:
            csv_logger = CSVLogger(save_name+'.log')
            check_point = multi_gpu_callback(self.network, save_name)
            callbacks = [csv_logger, check_point]
        else:
            callbacks = None

        #if n_gpus == 1:
        print("Training using a single GPU...")
        #history = self.network.fit_generator(self.generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        # else:
        #     print("Training using multiple GPUs...")
        #     parallel_model = multi_gpu_model(self.network, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
        #     parallel_model.compile(**self.compile_args)
        #     history = parallel_model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_data)

        #start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.batch_sz,1))
        fake = np.zeros((self.batch_sz,1))

        for epoch in range(epochs):
            for batch_i, (batch_img, batch_img_template) in enumerate(self.data_loader.load_batch()):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                phi = self.network.generator.predict([batch_img, batch_img_template])

                # deformable transformation
                transform = self.network.transformation.predict([batch_img, phi])

                # Create a ref image by perturbing th subject image with the template image
                perturbation_factor_alpha = 0.1 if epoch > epochs/2 else 0.2
                batch_ref = perturbation_factor_alpha * batch_img + (1- perturbation_factor_alpha) * batch_img_template

                # Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.network.discriminator.train_on_batch([batch_ref, batch_img], valid)
                d_loss_fake = self.network.discriminator.train_on_batch([transform, batch_img], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (z -> img is valid and img -> z is is invalid)
                g_loss = self.network.generator.train_on_batch([batch_img, batch_img_template], [valid])

                # Plot the progress
                print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))

                # # If at save interval => save generated image samples
                # if epoch % sample_interval == 0:
                #     self.sample_interval(epoch)

        #return history

    # def sample_images(self, epoch, batch_i):
    #     os.makedirs('ResultTest/', exist_ok=True)
    #
    #     idx, img_S = self.data_loader.load_data(batch_size=1, is_testing=True)
    #     template = self.data_loader.get_template()
    #
    #     phi = self.network.generator.predict([img_S, template])
    #     transform = self.network.transformation.predict([img_S, phi])
    #
    #     gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
    #
    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #
    #     titles = ['Original', 'Translated', 'Reconstructed']
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt])
    #             axs[i, j].set_title(titles[j])
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
    #     plt.close()



class multi_gpu_callback(Callback):
    """
    set callbacks for multi-gpu training
    """

    def __init__(self, model, save_name):
        super().__init__()
        self.model_to_save = model
        self.save_name = save_name

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save("{}_{}.h5".format(self.save_name, epoch))



if __name__ == '__main__':
    gan = GANUnetTrain()
    K.tensorflow_backend._get_available_gpus()
    gan.train_network()
