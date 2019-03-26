
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, CSVLogger

from ImageRegistrationGANs.data_loader import DataLoader
from ImageRegistrationGANs.GAN_unet_model40 import GANUnetModel40

import os
import datetime
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
        valid = np.ones((self.batch_sz,) + self.network.disc_patch)
        fake = np.zeros((self.batch_sz,) + self.network.disc_patch)

        start_time = datetime.datetime.now()
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
                g_loss = self.network.combined.train_on_batch([batch_img, batch_img_template, batch_ref], valid)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss,
                                                                        elapsed_time))


                # # If at save interval => save generated image samples
                # if epoch % sample_interval == 0:
                #     self.sample_interval(epoch)

        #return history


    def test_network(self):

        test_img = self.data_loader.imgs_test[0] # subject 16
        test_mask = self.data_loader.masks_test[0]
        test_img = np.float32(test_img)
        test_img = test_img * test_mask

        templ_img = self.data_loader.img_template
        templ_mask = self.data_loader.mask_template
        templ_img = templ_img * templ_mask

        input_sz = (40, 40, 40)
        step = (15, 15, 15)

        predict_img = np.zeros_like(test_img.shape, dtype = test_img.dtype)

        gap = (int((input_sz[0] - step[0]) / 2), int((input_sz[1] - step[1]) / 2), int((input_sz[2] - step[2]) / 2))
        for row in range(0, test_img.shape[0] - input_sz[0], step[0]):
            for col in range(0, test_img.shape[1] - input_sz[1], step[1]):
                for vol in range(0, test_img.shape[2] - input_sz[2], step[2]):

                    patch_sub_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=test_img.dtype)
                    patch_templ_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=test_img.dtype)

                    patch_sub_img[0, :, :, :, 0] = test_img[row:row + input_sz[0], col:col + input_sz[1], vol:vol + input_sz[2]]
                    patch_templ_img[0, :, :, :, 0] = templ_img[row:row + input_sz[0], col:col + input_sz[1], vol:vol + input_sz[2]]

                    patch_predict_phi = self.network.generator.predict([patch_sub_img,patch_templ_img] )
                    patch_predict_warped = self.network.transformation([patch_sub_img, patch_predict_phi])

                    predict_img[row + gap[0]:row + gap[0] + step[0], col + gap[1]:col + gap[1] + step[1], vol + gap[2]:vol + gap[2] + step[2]] = patch_predict_warped[0, :, :, :, 0]

        return predict_img

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

    def write_nifti(self,path, image_data, meta_dict={}):
        #    image_data = _standardize_axis_order(image_data, 'nii') # possibly a bad idea
        import nibabel as nib
        image = nib.Nifti1Image(image_data, None)
        for key in meta_dict.keys():
            if key in image.header.keys():
                image.header[key] = meta_dict[key]
        nib.save(image, path)


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

    predicted_image = gan.test_network()
    gan.write_nifti('/groups/scicompsoft/home/elmalakis/Work/Janelia/ImageRegistration/data/for_salma/preprocess_to_4/predicted_image', predicted_image)
