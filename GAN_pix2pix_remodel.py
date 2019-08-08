from __future__ import print_function, division

from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf

from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dropout, Concatenate, Cropping3D, Add
from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.models import Model

import numpy as np
import datetime
import nrrd
import os


# locally
#from ImageRegistrationGANs.helpers import dense_image_warp_3D, numerical_gradient_3D
#from ImageRegistrationGANs.data_loader import DataLoader
from data_loader import DataLoader   #To run on the cluster
from helpers import dense_image_warp_3D, numerical_gradient_3D, make_parallel #To run on the cluster'

__author__ = 'elmalakis'


class GAN_pix2pix():

    def __init__(self):

        K.set_image_data_format('channels_last')  # set format
        self.DEBUG = 1

        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.img_vols = 256
        self.channels = 1
        self.batch_sz = 1 # for testing locally to avoid memory allocation

        self.crop_size = (self.img_rows, self.img_cols, self.img_vols)

        self.img_shape = self.crop_size + (self.channels,)

        self.output_size = 256
        self.output_shape_g = self.crop_size + (3,)  # phi has three outputs. one for each X, Y, and Z dimensions
        self.input_shape_d =  self.crop_size  + (1,)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.output_size / 2 ** 4)
        self.output_shape_d = (patch, patch, patch,  self.channels)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32

        optimizer = Adam(0.001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        self.transformation = self.build_transformation()
        self.transformation.summary()

        # Input images and their conditioning images
        img_S = Input(shape=self.img_shape)
        img_T = Input(shape=self.img_shape)

        # Generate the deformable funtion
        phi = self.generator([img_S, img_T])
        # Transform S
        warped_S = self.transformation([img_S, phi])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        validity = self.discriminator([warped_S, img_T])

        self.combined = Model(inputs=[img_S, img_T], outputs=[validity, warped_S])
        self.combined.summary()

        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)


        if self.DEBUG:
            log_path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/logs_ganpix2pix_remod/'
            self.callback = TensorBoard(log_path)
            self.callback.set_model(self.combined)



        self.data_loader = DataLoader(batch_sz=self.batch_sz,
                                      crop_size=self.crop_size,
                                      dataset_name='fly',
                                      min_max=False,
                                      restricted_mask=False,
                                      use_hist_equilized_data=False,
                                      use_sharpen=False,
                                      use_golden=True)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0): # dropout is 50 ->change from the implementaion
            """Layers used during upsampling"""
            u = UpSampling3D(size=2)(layer_input)

            u = Conv3D(filters, kernel_size=f_size, padding='same')(u) # remove the strides
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Activation('relu')(u)
            u = Concatenate()([u, skip_input])
            return u


        img_S = Input(shape=self.img_shape, name='input_img_S')
        img_T = Input(shape=self.img_shape, name='input_img_T')

        d0 = Concatenate(axis=-1, name='combine_imgs_g')([img_S, img_T])
        #d0= Add(name='combine_imgs_g')([img_S, img_T])  #256

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)   #128
        d2 = conv2d(d1, self.gf*2)           #64
        d3 = conv2d(d2, self.gf*4)           #32
        d4 = conv2d(d3, self.gf*8)           #16
        d5 = conv2d(d4, self.gf*8)           #8
        d6 = conv2d(d5, self.gf*8)           #4
        d7 = conv2d(d6, self.gf*8)           #2

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)     #4
        u2 = deconv2d(u1, d5, self.gf*8)     #8
        u3 = deconv2d(u2, d4, self.gf*8)     #16
        u4 = deconv2d(u3, d3, self.gf*4)     #32
        u5 = deconv2d(u4, d2, self.gf*2)     #64
        u6 = deconv2d(u5, d1, self.gf)       #128

        u7 = UpSampling3D(size=2)(u6)        #256 #the original architecture from the paper is a bit different
        phi = Conv3D(filters=3, kernel_size=1, strides=1, padding='same')(u7) #256

        return  Model([img_S, img_T], outputs=phi, name='generator_model')


    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img_S = Input(shape=self.img_shape) #256 S
        img_T = Input(shape=self.img_shape) #256 T

        #img_T_cropped = Cropping3D(cropping=64)(img_T)  # 128

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_S, img_T])
        #combined_imgs = Add()([img_S, img_T])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', name='disc_sig')(d4) #original is linear activation no sigmoid

        return Model([img_S, img_T], validity, name='discriminator_model')


    def build_transformation(self):
        img_S = Input(shape=self.img_shape, name='input_img_S_transform')      # 256
        phi = Input(shape=self.output_shape_g, name='input_phi_transform')     # 256

        #img_S_cropped = Cropping3D(cropping=64)(img_S)  # 68x68x68

        warped_S = Lambda(dense_image_warp_3D, output_shape=self.input_shape_d)([img_S, phi])

        return Model([img_S, phi], warped_S,  name='transformation_layer')


    """
    Training
    """
    def train(self, epochs, batch_size=1, sample_interval=50):
        DEBUG =1
        path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
        os.makedirs(path+'generated_pix2pix/' , exist_ok=True)
        output_sz = 128
        input_sz = 256
        gap = int((input_sz - output_sz)/2)
        # Adversarial loss ground truths
        valid = np.ones((self.batch_sz,) + self.output_shape_d)
        fake = np.zeros((self.batch_sz,) + self.output_shape_d)

        start_time = datetime.datetime.now()
        for epoch in range(epochs):
            for batch_i, (batch_img, batch_img_template, batch_img_golden) in enumerate(self.data_loader.load_batch()):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Condition on B and generate a translate
                phi = self.generator.predict([batch_img, batch_img_template])
                transform = self.transformation.predict([batch_img, phi]) #256x256x256
                # Create a ref image by perturbing th subject image with the template image
                perturbation_factor_alpha = 0.1 if epoch > epochs/2 else 0.2
                batch_ref = perturbation_factor_alpha * batch_img + (1- perturbation_factor_alpha) * batch_img_template

                # batch_img_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_img.dtype)
                # batch_ref_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_ref.dtype)
                # batch_temp_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_img_template.dtype)
                # batch_golden_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_img_golden.dtype)

                # take only (128,128,128) from the (256,256,256) size
                # batch_img_sub[:, :, :, :, :] = batch_img[:, 0 + gap:0 + gap + output_sz,
                #                                             0 + gap:0 + gap + output_sz,
                #                                             0 + gap:0 + gap + output_sz, :]
                # batch_ref_sub[:, :, :, :, :] = batch_ref[:, 0 + gap:0 + gap + output_sz,
                #                                             0 + gap:0 + gap + output_sz,
                #                                             0 + gap:0 + gap + output_sz, :]
                # batch_golden_sub[:, :, :, :, :] = batch_img_golden[:, 0 + gap:0 + gap + output_sz,
                #                                             0 + gap:0 + gap + output_sz,
                #                                             0 + gap:0 + gap + output_sz, :]
                # batch_temp_sub[:, :, :, :, :] = batch_img_template[:, 0 + gap:0 + gap + output_sz,
                #                                                       0 + gap:0 + gap + output_sz,
                #                                                       0 + gap:0 + gap + output_sz, :]


                # Train the discriminators (original images = real / generated = Fake)
                # d_loss_real = self.discriminator.train_on_batch([batch_ref_sub, batch_img_template], valid)
                # d_loss_fake = self.discriminator.train_on_batch([transform, batch_img_template], fake)

                d_loss_real = self.discriminator.train_on_batch([batch_img_golden, batch_img_template], valid)
                d_loss_fake = self.discriminator.train_on_batch([transform, batch_img_template], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generator

                #g_loss = self.combined.train_on_batch([batch_img, batch_img_template], [valid, batch_golden_sub]) # The original implemntation has batch_img in the output
                g_loss = self.combined.train_on_batch([batch_img, batch_img_template], [valid, batch_img_golden])  # The original implemntation has batch_img in the output

                elapsed_time = datetime.datetime.now() - start_time

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss average: %f, acc average: %3d%%, D loss fake:%f, acc: %3d%%, D loss real: %f, acc: %3d%%] [G loss: %f]  time: %s"
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100 * d_loss[1],
                       d_loss_fake[0], 100 * d_loss_fake[1],
                       d_loss_real[0], 100 * d_loss_real[1],
                       g_loss[0],
                       elapsed_time))


                if self.DEBUG:
                    self.write_log(self.callback, ['g_loss'], [g_loss[0]], batch_i)
                    self.write_log(self.callback, ['d_loss'], [d_loss[0]], batch_i)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0 and epoch != 0 and epoch % 5 == 0:
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
        os.makedirs(path+'generated_pix2pix_remod/' , exist_ok=True)

        idx, imgs_S = self.data_loader.load_data(is_validation=True)
        imgs_T = self.data_loader.img_template

        predict_img = np.zeros(imgs_S.shape, dtype=imgs_S.dtype)

        input_sz = self.crop_size
        output_sz = (self.output_size, self.output_size, self.output_size)
        step = (24, 24, 24)

        gap = (int((input_sz[0] - output_sz[0]) / 2), int((input_sz[1] - output_sz[1]) / 2), int((input_sz[2] - output_sz[2]) / 2))
        start_time = datetime.datetime.now()

        for row in range(0, imgs_S.shape[0] - input_sz[0], step[0]):
            for col in range(0, imgs_S.shape[1] - input_sz[1], step[1]):
                for vol in range(0, imgs_S.shape[2] - input_sz[2], step[2]):

                    patch_sub_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=imgs_S.dtype)
                    patch_templ_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=imgs_T.dtype)

                    patch_sub_img[0, :, :, :, 0] = imgs_S[row:row + input_sz[0],
                                                         col:col + input_sz[1],
                                                         vol:vol + input_sz[2]]
                    patch_templ_img[0, :, :, :, 0] = imgs_T[row:row + input_sz[0],
                                                       col:col + input_sz[1],
                                                       vol:vol + input_sz[2]]

                    patch_predict_phi = self.generator.predict([patch_sub_img, patch_templ_img])
                    patch_predict_warped = self.transformation.predict([patch_sub_img, patch_predict_phi])

                    # predict_img[row + gap[0] :row + gap[0] + output_sz[0],
                    #             col + gap[1] :col + gap[1] + output_sz[1],
                    #             vol + gap[2] :vol + gap[2] + output_sz[2]] = patch_predict_warped[0, :, :, :, 0]

                    predict_img[row  :row + output_sz[0],
                                col  :col + output_sz[1],
                                vol  :vol + output_sz[2]] = patch_predict_warped[0, :, :, :, 0]


        elapsed_time = datetime.datetime.now() - start_time
        print(" --- Prediction time: %s" % (elapsed_time))

        nrrd.write(path+"generated_pix2pix_remod/%d_%d_%d" % (epoch, batch_i, idx), predict_img)

        file_name = 'gan_network' +str(epoch)
        # save the whole network
        gan.combined.save(path+'generated_pix2pix_remod/'+file_name + '.whole.h5', overwrite=True)
        print('Save the whole network to disk as a .whole.h5 file')
        model_jason = gan.combined.to_json()
        with open(path+'generated_pix2pix_remod/'+file_name + '_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.combined.save_weights(path+'generated_pix2pix_remod/'+file_name + '_weights.h5', overwrite=True)
        print('Save the network architecture in .json file and weights in .h5 file')

        # save the encoder network
        gan.generator.save(path+'generated_pix2pix_remod/'+file_name + '.gen.h5', overwrite=True)
        print('Save the generator network to disk as a .whole.h5 file')
        model_jason = gan.generator.to_json()
        with open(path+'generated_pix2pix_remod/'+file_name + '_gen_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.generator.save_weights(path+'generated_pix2pix_remod/'+file_name + '_gen_weights.h5', overwrite=True)
        print('Save the generator architecture in .json file and weights in .h5 file')


if __name__ == '__main__':
    # Use GPU
    K.tensorflow_backend._get_available_gpus()
    # launch tf debugger
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)

    gan = GAN_pix2pix()
    gan.train(epochs=20000, batch_size=1, sample_interval=200)





