from __future__ import print_function, division

import keras.backend as K
from tensorflow.python import debug as tf_debug

from keras.layers import BatchNormalization, Activation, MaxPooling3D, Cropping3D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate
from keras.layers import Lambda

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling3D, Conv3D

from keras.optimizers import Adam

from keras.models import Model

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import datetime
import os

from ImageRegistrationGANs.helpers import dense_image_warp_3D, numerical_gradient_3D
from ImageRegistrationGANs.data_loader import DataLoader


__author__ = 'elmalakis'


class GANUnetModel40():
    def __init__(self):
        self.img_row = 40
        self.img_col = 40
        self.img_depth = 40
        self.crop_size_g = (self.img_row, self.img_col, self.img_depth)
        self.crop_size_d = (self.img_row, self.img_col, self.img_depth)
        self.channels = 1
        self.input_shape_g = self.crop_size_g + (self.channels,)
        self.output_shape_g = self.crop_size_g + (3, ) # 3 dimensions (x, y, x)
        self.input_shape_d = self.crop_size_d + (self.channels,)

        # Calculate output shape of D
        patch = int(self.img_row / 2**2)  # 2 layers deep network
        self.disc_patch = (patch, patch, patch, 1)

        self.batch_sz = 8 # for testing locally
        self.data_loader = DataLoader(batch_sz=self.batch_sz)

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

        def d_layer(layer_input, filters, f_size=3, bn=True, name=''):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', name=name+'_conv3d')(layer_input)
            d = ReLU(name=name+'_relu')(d)
            if bn:
                d = BatchNormalization(name=name+'_bn')(d)
            return d

        img_A = Input(shape=self.input_shape_d, name='input_img_A') #40x40x40
        img_B = Input(shape=self.input_shape_d, name='input_img_B') #40x40x40

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1, name='combine_imgs_d')([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False, name='d1') #40x40x40
        d2 = d_layer(d1, self.df*2, name='d2')                    #40x40x40
        pool = MaxPooling3D(pool_size=(2, 2, 2), name='pool1_d')(d2)   #20x20x20

        d3 = d_layer(pool, self.df*4, name='d3')                  #20x20x20
        d4 = d_layer(d3, self.df*8, name='d4')                    #20x20x20
        pool = MaxPooling3D(pool_size=(2, 2, 2), name='pool2_d')(d4)   #10x10x10

        d4 = d_layer(pool, self.df*8, name='d4_2')                  #10x10x10

        # ToDo: check if the activation function 'sigmoid' is the right one or leave it to be linear; originally linear
        # ToDo: Use FC layer at the end like specified in the paper
        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid', name='validity')(d4) #10x10x10
        #validity = Dense(1, activation='sigmoid')(d4)

        return Model([img_A, img_B], validity, name='discriminator_model')


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
                    use_bias=False,
                    name=''):
            """
            3D convolutional layer (+ batch normalization) followed by ReLu activation
            """
            layer = Conv3D(filters=n_filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           use_bias=use_bias,
                           name=name+'_conv3d')(input_tensor)
            if batch_normalization:
                layer = BatchNormalization(name=name+'_bn')(layer)
            layer = Activation('relu', name=name+'_actrelu')(layer)

            return layer

        input_shape = self.input_shape_g
        img_S = Input(shape=input_shape, name='input_img_S')  # 40x40x40
        img_T = Input(shape=input_shape, name='input_img_T')  # 40x40x40

        # Concatenate subject image and template image by channels to produce input
        combined_imgs = Concatenate(axis=-1, name='combine_imgs_g')([img_S, img_T])

        # down-sampling
        down1 = g_layer(input_tensor=combined_imgs, n_filters=self.gf, padding='same', name='down1_1')  # 40x40x40
        down1 = g_layer(input_tensor=down1, n_filters=self.gf, padding='same', name='down1_2')  # 40x40x40
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='pool1')(down1)  # 20x20x20

        down2 = g_layer(input_tensor=pool1, n_filters=2 * self.gf, padding='same', name='down2_1')  # 20x20x20
        down2 = g_layer(input_tensor=down2, n_filters=2 * self.gf, padding='same', name='down2_2')  # 20x20x20
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), name='pool2')(down2)  # 10x10x10

        center = g_layer(input_tensor=pool2, n_filters=4 * self.gf, padding='same', name='center1_1')  # 10x10x10
        center = g_layer(input_tensor=center, n_filters=4 * self.gf, padding='same', name='center1_2')  # 10x10x10

        # up-sampling
        up2 = concatenate([down2, UpSampling3D(size=(2, 2, 2), name='up2_1')(center)])  # 20x20x20
        up2 = g_layer(input_tensor=up2, n_filters=2 * self.gf, padding='same', name='up2_2')  # 20x20x20
        up2 = g_layer(input_tensor=up2, n_filters=2 * self.gf, padding='same', name='up2_3')  # 20x20x20

        up1 = concatenate([down1, UpSampling3D(size=(2, 2, 2), name='up1_1')(up2)])  # 40x40x40
        up1 = g_layer(input_tensor=up1, n_filters=self.gf, padding='same', name='up1_2')  # 40x40x40
        up1 = g_layer(input_tensor=up1, n_filters=self.gf, padding='same', name='up1_3')  # 40x40x40

        # ToDo: check if the activation function 'sigmoid' is the right one or leave it to be linear; originally sigmoid
        phi = Conv3D(filters=3, kernel_size=(1, 1, 1), use_bias=False, name='phi')(up1)  # 40x40x40



        #warped_S = Lambda(dense_image_warp_3D)([img_S, phi])
        #model = Model([img_S, img_T], outputs=warped_S)

        model = Model([img_S, img_T], outputs=phi, name='generator_model')

        return model


    """
    Deformable Transformation Layer    
    """
    def build_transformation(self):
        img_S = Input(shape=self.input_shape_g, name='input_img_S_transform')  # 40x40x40
        phi = Input(shape=self.output_shape_g, name='input_phi_transform')    # 40x40x40

        #img_S_downsampled = Cropping3D(cropping=4)(MaxPooling3D(pool_size=(2,2,2))(img_S)) # 24x24x24
        # Put the warping function in a Lambda layer because it uses tensorflow
        warped_S = Lambda(dense_image_warp_3D, name='dense_image_warp')([img_S, phi]) # 40x40x40

        return Model([img_S, phi], warped_S, name='transform_layer_model')


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


    """
    Training
    """
    def train(self, epochs, batch_size=1, sample_interval=50):
        # Adversarial loss ground truths
        valid = np.ones((self.batch_sz,) + self.disc_patch)
        fake = np.zeros((self.batch_sz,) + self.disc_patch)

        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (batch_img, batch_img_template) in enumerate(self.data_loader.load_batch()):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                phi = self.generator.predict([batch_img, batch_img_template])

                # deformable transformation
                transform = self.transformation.predict([batch_img, phi])
                non_zero_indices = np.where(transform != 0)[0]
                print(len(non_zero_indices))

                # Create a ref image by perturbing th subject image with the template image
                perturbation_factor_alpha = 0.1 if epoch > epochs/2 else 0.2
                batch_ref = perturbation_factor_alpha * batch_img + (1- perturbation_factor_alpha) * batch_img_template

                # Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.discriminator.train_on_batch([batch_ref, batch_img], valid)
                d_loss_fake = self.discriminator.train_on_batch([transform, batch_img], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (z -> img is valid and img -> z is is invalid)
                g_loss = self.combined.train_on_batch([batch_img, batch_img_template, batch_ref], valid)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss,
                                                                        elapsed_time))


                # If at save interval => save generated image samples
            #if epoch % sample_interval == 0
            self.sample_images(epoch, batch_i) #save by the end of each epoch



    def sample_images(self, epoch, batch_i):
        os.makedirs('generated/' , exist_ok=True)

        idx, imgs_S, imgs_S_mask = self.data_loader.load_data(is_validation=True)
        imgs_T = self.data_loader.img_template
        imgs_T_mask = self.data_loader.mask_template

        imgs_S = imgs_S * imgs_S_mask
        imgs_T = imgs_T * imgs_T_mask

        predict_img = np.zeros(imgs_S.shape, dtype=imgs_S.dtype)

        input_sz = (40, 40, 40)
        step = (15, 15, 15)


        # No gap here because the ouput is the same as the input
        #gap = (int((input_sz[0] - step[0]) / 2), int((input_sz[1] - step[1]) / 2), int((input_sz[2] - step[2]) / 2))
        start_time = datetime.datetime.now()
        for row in range(0, imgs_S.shape[0] - input_sz[0], step[0]):
            for col in range(0, imgs_S.shape[1] - input_sz[1], step[1]):
                #for vol in range(0, imgs_S.shape[2] - input_sz[2], step[2]):

                    patch_sub_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=imgs_S.dtype)
                    patch_templ_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=imgs_T.dtype)

                    patch_sub_img[0, :, :, :, 0] = imgs_S[row:row + input_sz[0], col:col + input_sz[1], :]
                    patch_templ_img[0, :, :, :, 0] = imgs_T[row:row + input_sz[0], col:col + input_sz[1], :]

                    patch_predict_phi = self.generator.predict([patch_sub_img,patch_templ_img] )
                    patch_predict_warped = self.transformation.predict([patch_sub_img, patch_predict_phi])

                    #predict_img[row + gap[0]:row + gap[0] + step[0], col + gap[1]:col + gap[1] + step[1], vol+gap[2]:vol+gap[2]+step[2]] = patch_predict_warped[0, :, :, :, 0]
                    predict_img[row :row + input_sz[0], col :col + input_sz[1], : ] = patch_predict_warped[0, :, :, :, 0]

        elapsed_time = datetime.datetime.now() - start_time
        print(" --- Prediction time: %s" % (elapsed_time))

        self._write_nifti("generated/%d_%d" % (epoch, batch_i), predict_img)

        file_name = 'gan_network'

        # save the whole network
        gan.combined.save(file_name + '.whole.h5', overwrite=True)
        print('Save the whole network to disk as a .whole.h5 file')
        model_jason = gan.combined.to_json()
        with open(file_name + '_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.combined.save_weights(file_name + '_weights.h5', overwrite=True)
        print('Save the network architecture in .json file and weights in .h5 file')

        # save the generator network
        gan.generator.save(file_name + '.gen.h5', overwrite=True)
        print('Save the generator network to disk as a .whole.h5 file')
        model_jason = gan.combined.to_json()
        with open(file_name + '_gen_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.combined.save_weights(file_name + '_gen_weights.h5', overwrite=True)
        print('Save the generator architecture in .json file and weights in .h5 file')


    def _write_nifti(self,path, image_data, meta_dict={}):
        #    image_data = _standardize_axis_order(image_data, 'nii') # possibly a bad idea
        import nibabel as nib
        image = nib.Nifti1Image(image_data, None)
        for key in meta_dict.keys():
            if key in image.header.keys():
                image.header[key] = meta_dict[key]
        nib.save(image, path)


if __name__ == '__main__':
    # Use GPU
    K.tensorflow_backend._get_available_gpus()
    # launch tf debugger
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)

    gan = GANUnetModel40()
    gan.train(epochs=1000, batch_size=1, sample_interval=200)



