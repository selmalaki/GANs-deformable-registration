from __future__ import print_function, division

import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard

from keras_contrib.layers import InstanceNormalization

from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dropout, Concatenate

from keras.optimizers import Adam
from keras.models import Model

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import datetime
import nrrd
import os

from ImageRegistrationGANs.helpers import dense_image_warp_3D, numerical_gradient_3D
from ImageRegistrationGANs.data_loader import DataLoader

__author__ = 'elmalakis'


# Large amount of credit goes to:
# https://github.com/eriklindernoren/Keras-GAN#cyclegan
# which I've used as a reference for this implementation

class CycleGAN_model64():
    def __init__(self):

        self.DEBUG = 1

        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.img_depth = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_depth, self.channels)

        # Configure data loader
        self.dataset_name = 'fly'
        self.batch_sz = 16 # for testing locally to avoid memory allocation
        self.data_loader = DataLoader(batch_sz=self.batch_sz, dataset_name='fly', use_hist_equilized_data=True)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, patch,  1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)     # Subject
        img_B = Input(shape=self.img_shape)     # Template

        # Translate images to the other domain (Subject to Template)
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain (Reconstruct Subject)
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

        if self.DEBUG:
            log_path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/logs_cyclegan/'
            self.callback = TensorBoard(log_path)
            self.callback.set_model(self.combined)

    def build_generator(self):
        """U-Net Generator"""

        def conv3d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling3D(size=2)(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv3d(d0, self.gf)
        d2 = conv3d(d1, self.gf*2)
        d3 = conv3d(d2, self.gf*4)
        d4 = conv3d(d3, self.gf*8)

        # Upsampling
        u1 = deconv3d(d4, d3, self.gf*4)
        u2 = deconv3d(u1, d2, self.gf*2)
        u3 = deconv3d(u2, d1, self.gf)

        u4 = UpSampling3D(size=2)(u3)
        output_img = Conv3D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        #output_img = Conv3D(self.channels, kernel_size=1, strides=1, padding='valid', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, sample_interval=50):

        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((self.batch_sz,) + self.disc_patch)
        fake = np.zeros((self.batch_sz,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch()):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                if self.DEBUG:
                    self.write_log(self.callback, ['g_loss'], [g_loss[0]], batch_i)
                    self.write_log(self.callback, ['d_loss'], [d_loss[0]], batch_i)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)


    def write_log(self, callback, names, logs, batch_no):
        #https://github.com/eriklindernoren/Keras-GAN/issues/52
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()



    def sample_images(self, epoch, batch_i):
        path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
        os.makedirs(path+'generated_cyclegan/' , exist_ok=True)

        idx, img_S, img_S_mask = self.data_loader.load_data(is_validation=True)
        img_T = self.data_loader.img_template
        img_T_mask = self.data_loader.mask_template

        #img_S = img_S * img_S_mask   # img_A
        #img_T = img_T * img_T_mask   # img_B

        # img_S_mask = 1-img_S_mask # flip the mask to use np.ma.array
        # img_T_mask = 1-img_T_mask # flip the mask to use np.ma.array
        #
        # img_S_mask = np.uint8(img_S_mask)
        # img_S_masked = np.ma.array(img_S, mask=img_S_mask)
        # img_S = (img_S - img_S_masked.mean()) / img_S_masked.std()
        #
        # img_T_mask = np.uint8(img_T_mask)
        # img_T_masked = np.ma.array(img_T, mask=img_T_mask)
        # img_T = (img_T - img_T_masked.mean()) / img_T_masked.std()

        nrrd.write(path + "generated_cyclegan/originalS_%d_%d_%d" % (epoch, batch_i, idx), img_S)
        nrrd.write(path + "generated_cyclegan/originalT_%d_%d" % (epoch, batch_i), img_T)

        predict_img = np.zeros(img_S.shape, dtype=img_S.dtype)
        predict_templ = np.zeros(img_T.shape, dtype=img_T.dtype)

        input_sz = (64, 64, 64)
        step = (24, 24, 24)

        start_time = datetime.datetime.now()
        for row in range(0, img_S.shape[0] - input_sz[0], step[0]):
            for col in range(0, img_S.shape[1] - input_sz[1], step[1]):
                for vol in range(0, img_S.shape[2] - input_sz[2], step[2]):

                    patch_sub_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=img_S.dtype)
                    patch_templ_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=img_T.dtype)

                    patch_sub_img[0, :, :, :, 0] = img_S[row:row + input_sz[0],
                                                        col:col + input_sz[1],
                                                        vol:vol + input_sz[2]]
                    patch_templ_img[0, :, :, :, 0] = img_T[row:row + input_sz[0],
                                                     col:col + input_sz[1],
                                                     vol:vol + input_sz[2]]

                    # Translate images to the other domain
                    fake_B = self.g_AB.predict(patch_sub_img)   # subject to template
                    fake_A = self.g_BA.predict(patch_templ_img) # template to subject
                    # Translate back to original domain
                    #reconstr_A = self.g_BA.predict(fake_B)
                    #reconstr_B = self.g_AB.predict(fake_A)

                    predict_img[row:row + input_sz[0],
                                col:col + input_sz[1],
                                vol:vol + input_sz[2]] = fake_B[0, :, :, :, 0]
                    predict_templ[row:row + input_sz[0],
                                col:col + input_sz[1],
                                vol:vol + input_sz[2]] = fake_A[0, :, :, :, 0]

        elapsed_time = datetime.datetime.now() - start_time
        print(" --- Prediction time: %s" % (elapsed_time))

        nrrd.write(path+"generated_cyclegan/S2T_%d_%d_%d" % (epoch, batch_i, idx), predict_img)
        nrrd.write(path + "generated_cyclegan/T2S_%d_%d_%d" % (epoch, batch_i, idx), predict_templ)

        file_name = 'gan_network'
        # save the whole network
        gan.combined.save(path+'generated_cyclegan/'+file_name + '.whole.h5', overwrite=True)
        print('Save the whole network to disk as a .whole.h5 file')
        model_jason = gan.combined.to_json()
        with open(path+'generated_cyclegan/' + file_name+ '_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.combined.save_weights(path+'generated_cyclegan/' + file_name + '_weights.h5', overwrite=True)
        print('Save the network architecture in .json file and weights in .h5 file')

        # save the generator network the translation one
        gan.g_AB.save(path+'generated_cyclegan/' + file_name + '.gen.h5', overwrite=True)
        print('Save the generator network to disk as a .whole.h5 file')
        model_jason = gan.combined.to_json()
        with open(path+'generated_cyclegan/' + file_name + '_gen_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.combined.save_weights(path+'generated_cyclegan/' + file_name + '_gen_weights.h5', overwrite=True)
        print('Save the generator architecture in .json file and weights in .h5 file')



if __name__ == '__main__':
    # Use GPU
    K.tensorflow_backend._get_available_gpus()
    gan = CycleGAN_model64()
    gan.train(epochs=200, sample_interval=200)
