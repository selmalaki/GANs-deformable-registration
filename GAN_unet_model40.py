from __future__ import print_function, division

import keras.backend as K

from keras.layers import BatchNormalization, Activation, MaxPooling3D, Cropping3D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate
from keras.layers import Lambda

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling3D, Conv3D

from keras.optimizers import Adam

from keras.models import Model

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import scipy
import sys

from .helpers import dense_image_warp_3D, numerical_gradient_3D


__author__ = 'elmalakis'


class GANUnetModel40:
    def __init__(self):
        self.img_row = 40
        self.img_col = 40
        self.img_depth = 40
        self.crop_size_g = (self.img_row, self.img_col, self.img_depth)
        self.crop_size_d = (self.img_row, self.img_col, self.img_depth)
        self.channels = 1
        self.input_shape_g = self.crop_size_g + (self.channels,)
        self.input_shape_d = self.crop_size_d + (self.channels,)

        # Calculate output shape of D
        patch = int(self.img_row / 2**2)  # 2 layers deep network
        self.disc_patch = (patch, patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        print('--- Discriminator model ---')
        self.discriminator.summary()


        # Build the generator
        self.generator = self.build_generator()
        print('--- Generator model ---')
        self.generator.summary()

        # Build the deformable transformation layer
        self.transformation = self.build_transformation()
        print('--- Transformation layer ---')
        self.transformation.summary()

        # Input images
        img_S = Input(shape=self.input_shape_g) # subject image S
        img_T = Input(shape=self.input_shape_g) # template image T
        img_R = Input(shape=self.input_shape_d) # reference image R

        # By conditioning on T generate a warped transformation function of S
        phi = self.generator([img_S, img_T])

        # Transform S
        warped_S = self.transformation([img_S, phi])

        # Use Python partial to provide loss function with additional deformable field argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  phi=phi)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.transformation.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([warped_S, img_R])

        self.combined = Model(inputs=[img_S, img_T, img_R], outputs=valid)
        self.combined.compile(loss=partial_gp_loss,
                              optimizer=optimizer)


    """
    Discriminator Network
    """
    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=3, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = ReLU()(d)
            if bn:
                d = BatchNormalization()(d)
            return d

        img_A = Input(shape=self.input_shape_d) #40x40x40
        img_B = Input(shape=self.input_shape_d) #40x40x40

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False) #40x40x40
        d2 = d_layer(d1, self.df*2)                    #40x40x40
        pool = MaxPooling3D(pool_size=(2, 2, 2))(d2)   #20x20x20

        d3 = d_layer(pool, self.df*4)                  #20x20x20
        d4 = d_layer(d3, self.df*8)                    #20x20x20
        pool = MaxPooling3D(pool_size=(2, 2, 2))(d4)   #10x10x10

        d4 = d_layer(pool, self.df*8)                  #10x10x10

        # ToDo: check if the activation function 'sigmoid' is the right one or leave it to be linear; originally linear
        # ToDo: Use FC layer at the end like specified in the paper
        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4) #10x10x10
        #validity = Dense(1, activation='sigmoid')(d4)

        return Model([img_A, img_B], validity)


    """
     Generator Network
     """
    def build_generator(self):
        """U-Net Generator"""

        def g_layer(input_tensor,
                    n_filters,
                    kernel_size=(3, 3, 3),
                    batch_normalization=True,
                    scale=True,
                    padding='valid',
                    use_bias=False):
            """
            3D convolutional layer (+ batch normalization) followed by ReLu activation
            """
            layer = Conv3D(filters=n_filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           use_bias=use_bias)(input_tensor)
            if batch_normalization:
                layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)

            return layer

        input_shape = self.input_shape_g
        img_S = Input(shape=input_shape)  # 40x40x40
        img_T = Input(shape=input_shape)  # 40x40x40

        # Concatenate subject image and template image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_S, img_T])

        # down-sampling
        down1 = g_layer(input_tensor=combined_imgs, n_filters=self.gf, padding='same')  # 40x40x40
        down1 = g_layer(input_tensor=down1, n_filters=self.gf, padding='same')  # 40x40x40
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(down1)  # 20x20x20

        down2 = g_layer(input_tensor=pool1, n_filters=2 * self.gf, padding='same')  # 20x20x20
        down2 = g_layer(input_tensor=down2, n_filters=2 * self.gf, padding='same')  # 20x20x20
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(down2)  # 10x10x10

        center = g_layer(input_tensor=pool2, n_filters=4 * self.gf, padding='same')  # 10x10x10
        center = g_layer(input_tensor=center, n_filters=4 * self.gf, padding='same')  # 10x10x10

        # up-sampling
        up2 = concatenate([down2, UpSampling3D(size=(2, 2, 2))(center)])  # 20x20x20
        up2 = g_layer(input_tensor=up2, n_filters=2 * self.gf, padding='same')  # 20x20x20
        up2 = g_layer(input_tensor=up2, n_filters=2 * self.gf, padding='same')  # 20x20x20

        up1 = concatenate([down1, UpSampling3D(size=(2, 2, 2))(up2)])  # 40x40x40
        up1 = g_layer(input_tensor=up1, n_filters=self.gf, padding='same')  # 40x40x40
        up1 = g_layer(input_tensor=up1, n_filters=self.gf, padding='same')  # 40x40x40

        # ToDo: check if the activation function 'sigmoid' is the right one or leave it to be linear; originally sigmoid
        phi = Conv3D(filters=1, kernel_size=(1, 1, 1), use_bias=False)(up1)  # 40x4040

        #warped_S = Lambda(dense_image_warp_3D)([img_S, phi])
        #model = Model([img_S, img_T], outputs=warped_S)

        model = Model([img_S, img_T], outputs=phi)

        return model


    """
    Deformable Transformation Layer    
    """
    def build_transformation(self):
        img_S = Input(shape=self.input_shape_g)  # 40x40x40
        phi = Input(shape=self.input_shape_d)    # 40x40x40

        #img_S_downsampled = Cropping3D(cropping=4)(MaxPooling3D(pool_size=(2,2,2))(img_S)) # 24x24x24
        # Put the warping function in a Lambda layer because it uses tensorflow
        warped_S = Lambda(dense_image_warp_3D)([img_S, phi]) # 40x40x40

        return Model([img_S, phi], warped_S)


    """
    Define losses
    """
    def gradient_penalty_loss(self, y_true, y_pred, phi):
        """
        Computes gradient penalty on phi to ensure smoothness
        """
        if y_true == 0:
            lr = -K.log(1-y_pred) # negative sign because the loss should be a positive value
        else:
            lr = 0  # no loss in the other case because the y_true in all the generation case should be 0

        # compute the numerical gradient of phi
        gradients = numerical_gradient_3D(phi)

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty) + lr





