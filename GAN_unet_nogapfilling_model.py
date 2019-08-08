from __future__ import print_function, division

from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf

from keras.layers import BatchNormalization, Activation, MaxPooling3D, Cropping3D
from keras.layers import Input, Concatenate, concatenate, Reshape, Add
#from keras.layers import Lambda
from keras.layers.core import Flatten, Dense, Lambda

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling3D, Conv3D, Conv3DTranspose

from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import datetime
import nrrd
import os

#from ImageRegistrationGANs.helpers import dense_image_warp_3D, numerical_gradient_3D
#from ImageRegistrationGANs.data_loader import DataLoader


from data_loader import DataLoader   #To run on the cluster'
from helpers import dense_image_warp_3D, numerical_gradient_3D #To run on the cluster'


__author__ = 'elmalakis'


class GANUnetNoGapFillingModel():

    def __init__(self):

        K.set_image_data_format('channels_last')  # set format
        K.set_image_dim_ordering('tf')
        self.DEBUG = 1

        # Input shape
        self.img_rows = 192
        self.img_cols = 192
        self.img_vols = 192
        self.channels = 1
        self.batch_sz = 1  # for testing locally to avoid memory allocation

        self.crop_size = (self.img_rows, self.img_cols, self.img_vols)

        self.img_shape = self.crop_size + (self.channels,)

        self.output_size = 192
        self.output_shape_g = self.crop_size + (3,)  # phi has three outputs. one for each X, Y, and Z dimensions
        self.input_shape_d = self.crop_size + (1,)

        # Calculate output shape of D
        patch = int(self.output_size / 2 ** 4)
        self.output_shape_d = (patch, patch, patch,  self.channels)

        self.batch_sz = 1 # for testing locally to avoid memory allocation

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32

        # Train the discriminator faster than the generator
        #optimizerD = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # in the paper the learning rate is 0.001 and weight decay is 0.5
        optimizerD = Adam(0.001, decay=0.05)  # in the paper the decay after 50K iterations by 0.5
        self.decay = 0.5
        self.iterations_decay = 50
        self.learning_rate = 0.001
        #optimizerG = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # in the paper the decay after 50K iterations by 0.5
        optimizerG = Adam(0.001, decay=0.05)  # in the paper the decay after 50K iterations by 0.5

        # Build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
                                    optimizer=optimizerD,
                                    metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # Build the deformable transformation layer
        self.transformation = self.build_transformation()
        self.transformation.summary()

        # Input images and their conditioning images
        img_S = Input(shape=self.img_shape)
        img_T = Input(shape=self.img_shape)

        # By conditioning on T generate a warped transformation function of S
        phi = self.generator([img_S, img_T])

        # Transform S
        warped_S = self.transformation([img_S, phi])

        # Use Python partial to provide loss function with additional deformable field argument
        partial_gp_loss = partial(self.gradient_penalty_loss, phi=phi)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        validity = self.discriminator([warped_S, img_T])

        self.combined = Model(inputs=[img_S, img_T], outputs=validity)
        self.combined.summary()
        self.combined.compile(loss = partial_gp_loss, optimizer=optimizerG)

        if self.DEBUG:
            log_path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/logs_ganunet_nogap/'
            os.makedirs(log_path, exist_ok=True)
            self.callback = TensorBoard(log_path)
            self.callback.set_model(self.combined)

        self.data_loader = DataLoader(batch_sz=self.batch_sz,
                                      crop_size=self.crop_size,
                                      dataset_name='fly',
                                      min_max=False,
                                      restricted_mask=False,
                                      use_hist_equilized_data=False,
                                      use_golden=False)

    """
    Generator Network
    """
    def build_generator(self):
        """U-Net Generator"""
        def conv3d(input_tensor,
                        n_filters,
                        kernel_size=(3, 3, 3),
                        batch_normalization=True,
                        scale=True,
                        padding='valid',
                        use_bias=False,
                        name=''):
            """
            3D convolutional layer (+ batch normalization) followed by ReLu activation
            """
            layer = Conv3D(filters=n_filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           use_bias=use_bias,
                           name=name + '_conv3d')(input_tensor)
            if batch_normalization:
                layer = BatchNormalization(momentum=0.8, name=name+'_bn', scale=scale)(layer)
            #layer = LeakyReLU(alpha=0.2, name=name + '_actleakyrelu')(layer)
            layer = Activation("relu")(layer)
            return layer


        def deconv3d(input_tensor,
                        n_filters,
                        kernel_size=(3, 3, 3),
                        batch_normalization=True,
                        scale=True,
                        padding='valid',
                        use_bias=False,
                        name=''):
            """
            3D deconvolutional layer (+ batch normalization) followed by ReLu activation
            """
            layer = UpSampling3D(size=2)(input_tensor)
            layer = Conv3D(filters=n_filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           use_bias=use_bias,
                           name=name + '_conv3d')(layer)

            if batch_normalization:
                layer = BatchNormalization(momentum=0.8, name=name+'_bn', scale=scale)(layer)
            #layer = LeakyReLU(alpha=0.2, name=name + '_actleakyrelu')(layer)
            layer = Activation("relu")(layer)
            return layer

        img_S = Input(shape=self.img_shape, name='input_img_S')
        img_T = Input(shape=self.img_shape, name='input_img_T')

        combined_imgs = Add(name='combine_imgs_g')([img_S,img_T])

        # downsampling
        down1 = conv3d(input_tensor=combined_imgs, n_filters=self.gf, padding='same', name='down1_1')  # 192
        down1 = conv3d(input_tensor=down1, n_filters=self.gf, padding='same', name='down1_2')          # 192
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='pool1')(down1)                                 # 96

        down2 = conv3d(input_tensor=pool1, n_filters=2 * self.gf, padding='same', name='down2_1')      # 96
        down2 = conv3d(input_tensor=down2, n_filters=2 * self.gf, padding='same', name='down2_2')      # 96
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), name='pool2')(down2)                                 # 48

        down3 = conv3d(input_tensor=pool2, n_filters=4 * self.gf, padding='same', name='down3_1')      # 48
        down3 = conv3d(input_tensor=down3, n_filters=4 * self.gf, padding='same', name='down3_2')      # 48
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), name='pool3')(down3)                                 #24

        down4 = conv3d(input_tensor=pool3, n_filters=8 * self.gf, padding='same', name='down4_1')      # 24
        down4 = conv3d(input_tensor=down4, n_filters=8 * self.gf, padding='same', name='down4_2')      # 24
        pool4 = MaxPooling3D(pool_size=(2, 2, 2), name='pool4')(down4)                                 # 12

        down5 = conv3d(input_tensor=pool4, n_filters=8 * self.gf, padding='same', name='down5_1')      # 12
        down5 = conv3d(input_tensor=down5, n_filters=8 * self.gf, padding='same', name='down5_2')      # 12
        pool5 = MaxPooling3D(pool_size=(2, 2, 2), name='pool5')(down5)                                  # 6

        center = conv3d(input_tensor=pool5, n_filters=16 * self.gf, padding='same', name='center1')     # 6
        center = conv3d(input_tensor=center, n_filters=16 * self.gf, padding='same', name='center2')    # 6

        # upsampling
        up5 = deconv3d(input_tensor=center, n_filters = 8*self.gf, padding='same', name='up5')          # 12
        up5 = concatenate([up5,down5])                                                                  # 12
        up5 = conv3d(input_tensor=up5, n_filters=8 * self.gf, padding='same', name='up5_1')             # 12
        up5 = conv3d(input_tensor=up5, n_filters=8 * self.gf, padding='same', name='up5_2')             # 12

        up4 = deconv3d(input_tensor=up5, n_filters=8 * self.gf, padding='same', name='up4')             #24
        up4 = concatenate([up4, down4])                                                                 # 24
        up4 = conv3d(input_tensor=up4, n_filters=8 * self.gf, padding='same', name='up4_1')             # 24
        up4 = conv3d(input_tensor=up4, n_filters=8 * self.gf, padding='same', name='up4_2')             # 24

        up3 = deconv3d(input_tensor=up4, n_filters=4 * self.gf, padding='same', name='up3')             #48
        up3 = concatenate([up3, down3])                                                                 # 48
        up3 = conv3d(input_tensor=up3, n_filters=4 * self.gf, padding='same', name='up3_1')            # 48
        up3 = conv3d(input_tensor=up3, n_filters=4 * self.gf, padding='same', name='up3_2')            # 48

        up2 = deconv3d(input_tensor=up3, n_filters=2 * self.gf, padding='same', name='up2')             # 96
        up2 = concatenate([up2, down2])                                                                # 96
        up2 = conv3d(input_tensor=up2, n_filters=2 * self.gf, padding='same', name='up2_1')            # 96
        up2 = conv3d(input_tensor=up2, n_filters=2 * self.gf, padding='same', name='up2_2')            # 96

        up1 = deconv3d(input_tensor=up2, n_filters=self.gf, padding='same', name='up1')                 # 192
        up1 = concatenate([up1, down1])                                                                 # 192
        up1 = conv3d(input_tensor=up1, n_filters=self.gf, padding='same', name='up1_1')                # 192
        up1 = conv3d(input_tensor=up1, n_filters=self.gf, padding='same', name='up1_2')                # 192

        phi = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=1, use_bias=False, padding='same', name='phi')(up1)                 #192

        model = Model([img_S, img_T], outputs=phi, name='generator_model')

        return model

    """
    Discriminator Network
    """
    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            #d = LeakyReLU(alpha=0.2)(d)
            d = Activation("relu")(d)
            return d

        img_S = Input(shape=self.img_shape) #192 S
        img_T = Input(shape=self.img_shape) #192 T

        combined_imgs = Add()([img_S, img_T])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid', name='disc_sig')(d4)

        return Model([img_S, img_T], validity, name='discriminator_model')

    """
    Transformation Network
    """
    def build_transformation(self):
        img_S = Input(shape=self.img_shape, name='input_img_S_transform')      # 192
        phi = Input(shape=self.output_shape_g, name='input_phi_transform')     # 192

        warped_S = Lambda(dense_image_warp_3D, output_shape=self.input_shape_d)([img_S, phi])

        return Model([img_S, phi], warped_S,  name='transformation_layer')


    """
    Define losses
    """
    def gradient_penalty_loss(self, y_true, y_pred, phi):
        """
        Computes gradient penalty on phi to ensure smoothness
        """
        lr = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        # compute the numerical gradient of phi
        gradients = numerical_gradient_3D(phi)
        # #if self.DEBUG: gradients = K.print_tensor(gradients, message='gradients are:')
        #
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # # compute lambda * (1 - ||grad||)^2 still for each single sample
        # #gradient_penalty = K.square(1 - gradient_l2_norm)
        # # return the mean as loss over all the batch samples
        return K.mean(gradient_l2_norm) + lr
        #return gradients_sqr_sum + lr


    """
    Training
    """
    def train(self, epochs, batch_size=1, sample_interval=50):

        # Adversarial loss ground truths
        # hard labels
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
                transform = self.transformation.predict([batch_img, phi])  # 256x256x256
                # Create a ref image by perturbing th subject image with the template image
                perturbation_factor_alpha = 0.1 if epoch > epochs / 2 else 0.2
                batch_ref = perturbation_factor_alpha * batch_img + (1 - perturbation_factor_alpha) * batch_img_template

                d_loss_real = self.discriminator.train_on_batch([batch_ref, batch_img_template], valid)
                d_loss_fake = self.discriminator.train_on_batch([transform, batch_img_template], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                g_loss = self.combined.train_on_batch([batch_img, batch_img_template], valid)

                elapsed_time = datetime.datetime.now() - start_time

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss average: %f, acc average: %3d%%, D loss fake:%f, acc: %3d%%, D loss real: %f, acc: %3d%%] [G loss: %f]  time: %s"
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100 * d_loss[1],
                       d_loss_fake[0], 100 * d_loss_fake[1],
                       d_loss_real[0], 100 * d_loss_real[1],
                       g_loss,
                       elapsed_time))

                if self.DEBUG:
                    self.write_log(self.callback, ['g_loss'], [g_loss], batch_i)
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
        os.makedirs(path+'generated_unet_nogap/' , exist_ok=True)

        idx, imgs_S = self.data_loader.load_data(is_validation=True)
        imgs_T = self.data_loader.img_template

        predict_img = np.zeros(imgs_S.shape, dtype=imgs_S.dtype)
        predict_phi = np.zeros(imgs_S.shape + (3,), dtype=imgs_S.dtype)

        input_sz = self.crop_size
        output_sz = (self.output_size, self.output_size, self.output_size)
        step = (64, 64, 64)

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

                    predict_img[row:row + output_sz[0],
                                col:col + output_sz[1],
                                vol:vol + output_sz[2]] = patch_predict_warped[0, :, :, :, 0]
                    predict_phi[row :row  + output_sz[0],
                               col :col  + output_sz[1],
                               vol :vol  + output_sz[2],:] = patch_predict_phi[0, :, :, :, :]

        elapsed_time = datetime.datetime.now() - start_time
        print(" --- Prediction time: %s" % (elapsed_time))

        nrrd.write(path+"generated_unet_nogap/%d_%d_%d" % (epoch, batch_i, idx), predict_img)
        self.data_loader._write_nifti(path+"generated_unet_nogap/phi%d_%d_%d" % (epoch, batch_i, idx), predict_phi)

        if epoch%10 == 0:
            file_name = 'gan_network'+ str(epoch)
            # save the whole network
            self.combined.save(path+ 'generated_unet_nogap/'+ file_name + '.whole.h5', overwrite=True)
            print('Save the whole network to disk as a .whole.h5 file')
            model_jason = self.combined.to_json()
            with open(path+ 'generated_unet_nogap/'+file_name + '_arch.json', 'w') as json_file:
                json_file.write(model_jason)
                self.combined.save_weights(path+ 'generated_unet_nogap/'+file_name + '_weights.h5', overwrite=True)
            print('Save the network architecture in .json file and weights in .h5 file')

            # # save the generator network
            # self.generator.save(path+ 'generated_unet_nogap/'+file_name + '.gen.h5', overwrite=True)
            # print('Save the generator network to disk as a .whole.h5 file')
            # model_jason = self.generator.to_json()
            # with open(path+ 'generated_unet_nogap/'+file_name + '_gen_arch.json', 'w') as json_file:
            #     json_file.write(model_jason)
            #     self.generator.save_weights(path+ 'generated_unet_nogap/'+file_name + '_gen_weights.h5', overwrite=True)
            # print('Save the generator architecture in .json file and weights in .h5 file')


if __name__ == '__main__':
    # Use GPU
    K.tensorflow_backend._get_available_gpus()
    K.image_dim_ordering()
    K.set_image_dim_ordering('tf')
    gan = GANUnetNoGapFillingModel()
    gan.train(epochs=20000, batch_size=4, sample_interval=200)





