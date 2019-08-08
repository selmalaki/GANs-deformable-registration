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

from functools import partial
import numpy as np
import datetime
import nrrd
import os

# locally
#from ImageRegistrationGANs.helpers import dense_image_warp_3D, numerical_gradient_3D
#from ImageRegistrationGANs.data_loader import DataLoader

from data_loader import DataLoader   #To run on the cluster'
from helpers import dense_image_warp_3D, numerical_gradient_3D, make_parallel #To run on the cluster'


__author__ = 'elmalakis'


class GANUnetModel148():

    def __init__(self):

        K.set_image_data_format('channels_last')  # set format
        K.set_image_dim_ordering('tf')
        self.DEBUG = 1

        self.crop_size_g = (148, 148, 148)
        self.crop_size_d = (60, 60, 60)

        self.channels = 1

        self.input_shape_g = self.crop_size_g + (self.channels,)
        self.input_shape_d = self.crop_size_d + (self.channels,)

        self.output_shape_g = (60, 60, 60) + (3,)  # phi has three outputs. one for each X, Y, and Z dimensions
        self.output_shape_d = (15, 15, 15) + (self.channels,)
        #self.output_shape_d_v2 = (5, 5, 5) + (self.channels,)

        self.batch_sz = 1 # for testing locally to avoid memory allocation

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32

        #optimizerD = Adam(0.001, decay=0.05) # in the paper the learning rate is 0.001 and weight decay is 0.5
        optimizerD = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.decay = 0.5
        self.iterations_decay = 50
        self.learning_rate = 0.001
        #optimizerG = Adam(0.001, decay=0.05) # in the paper the decay after 50K iterations by 0.5
        optimizerG = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        # Build the three networks
        self.generator = self.build_generator()
        self.generator.summary()
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.transformation = self.build_transformation()
        self.transformation.summary()

        # Compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                    optimizer=optimizerD,
                                    metrics=['accuracy'])

        # Build the generator
        img_S = Input(shape=self.input_shape_g) # subject image S
        img_T = Input(shape=self.input_shape_g) # template image T

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
            log_path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/logs_ganunet_v148/'
            self.callback = TensorBoard(log_path)
            self.callback.set_model(self.combined)

        self.data_loader = DataLoader(batch_sz=self.batch_sz,
                                      crop_size=self.crop_size_g,
                                      dataset_name='fly')


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
            # if batch_normalization:
            #     layer = BatchNormalization(name=name+'_bn')(layer)
            #layer = Activation('relu', name=name+'_actrelu')(layer)
            # Add BN after activation
            if batch_normalization:
                layer = BatchNormalization(momentum=0.8, name=name+'_bn', scale=scale)(layer)
            layer = LeakyReLU(alpha=0.2, name=name + '_actleakyrelu')(layer)
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
            # BN before activation
            if batch_normalization:
                layer = BatchNormalization(momentum=0.8, name=name+'_bn', scale=scale)(layer)
            layer = LeakyReLU(alpha=0.2, name=name + '_actleakyrelu')(layer)
            return layer

        img_S = Input(shape=self.input_shape_g, name='input_img_S')                                            # 148x148x148
        img_T = Input(shape=self.input_shape_g, name='input_img_T')                                            # 148x148x148

        # Concatenate subject image and template image by channels to produce input
        #combined_imgs = Concatenate(axis=-1, name='combine_imgs_g')([img_S, img_T])
        combined_imgs = Add(name='combine_imgs_g')([img_S, img_T])


        # downsampling
        down1 = conv3d(input_tensor=combined_imgs, n_filters=self.gf, padding='valid', name='down1_1') #146
        down1 = conv3d(input_tensor=down1, n_filters=self.gf, padding='valid', name='down1_2')         #144
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='pool1')(down1)                                 #72

        down2 = conv3d(input_tensor=pool1, n_filters=2 * self.gf, padding='valid', name='down2_1')     #70
        down2 = conv3d(input_tensor=down2, n_filters=2 * self.gf, padding='valid', name='down2_2')     #68
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), name='pool2')(down2)                                 #34

        down3 = conv3d(input_tensor=pool2, n_filters=4 * self.gf, padding='valid', name='down3_1')     #32
        down3 = conv3d(input_tensor=down3, n_filters=4 * self.gf, padding='valid', name='down3_2')     #30
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), name='pool3')(down3)                                 #15

        center = conv3d(input_tensor=pool3, n_filters=8 * self.gf, padding='valid', name='center1')    #13
        center = conv3d(input_tensor=center, n_filters=8 * self.gf, padding='valid', name='center2')   #11

        # upsampling with gap filling
        up3 = deconv3d(input_tensor=center, n_filters = 4*self.gf, padding='same', name='up3')         #22
        gap3 = conv3d(input_tensor=down3, n_filters=4*self.gf, padding='valid', name='gap3_1')         #28
        gap3 = conv3d(input_tensor=gap3, n_filters=4*self.gf, padding='valid', name='gap3_2')          #26
        up3 = concatenate([Cropping3D(2)(gap3), up3], name='up3concat')                                #22
        up3 = conv3d(input_tensor=up3, n_filters=4*self.gf, padding='valid', name='up3conv_1')         #20
        up3 = conv3d(input_tensor=up3, n_filters=4*self.gf, padding='valid', name='up3conv_2')         #18

        up2 = deconv3d(input_tensor=up3, n_filters=2 * self.gf, padding='same', name='up2')            #36
        gap2 = conv3d(input_tensor=down2, n_filters=2 * self.gf, padding='valid', name='gap2_1')       #66
        for i in range(2, 7):
            gap2 = conv3d(input_tensor=gap2, n_filters=2 * self.gf, padding='valid', name='gap2_'+str(i))        #56

        up2 = concatenate([Cropping3D(10)(gap2), up2], name='up2concat')                               #36
        up2 = conv3d(input_tensor=up2, n_filters=2 * self.gf, padding='valid', name='up2conv_1')       #34
        up2 = conv3d(input_tensor=up2, n_filters=2 * self.gf, padding='valid', name='up2conv_2')       #32


        up1 = deconv3d(input_tensor=up2, n_filters=self.gf, padding='same', name='up1')                #64
        gap1 = conv3d(input_tensor=down1, n_filters=self.gf, padding='valid', name='gap1_1')           #142
        for i in range(2,21):
            gap1 = conv3d(input_tensor=gap1, n_filters=self.gf, padding='valid', name='gap1_'+str(i))          #104
        up1 = concatenate([Cropping3D(20)(gap1), up1], name='up1concat')                               #64
        up1 = conv3d(input_tensor=up1, n_filters=self.gf, padding='valid', name='up1conv_1')           #62
        up1 = conv3d(input_tensor=up1, n_filters=self.gf, padding='valid', name='up1conv_2')           #60

        phi = Conv3D(filters=3, kernel_size=(1, 1, 1), padding='same', use_bias=False, name='phi')(up1) #60

        model = Model([img_S, img_T], outputs=phi, name='generator_model')

        return model

    """
    Discriminator Network
    """
    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=5, bn=True, scale=True,  name=''): #change the bn to False
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', name=name+'_conv3d')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8, name=name+'_bn', scale=scale)(d)
            d = LeakyReLU(alpha=0.2, name=name + '_leakyrelu')(d)
            return d

        img_S = Input(shape=self.input_shape_d, name='input_img_A')             # 60 warped_img or reference
        img_T = Input(shape=self.input_shape_g, name='input_img_T')             # 148 template

        img_T_cropped = Cropping3D(cropping=44)(img_T)                          # 60

        # Concatenate image and conditioning image by channels to produce input
        #combined_imgs = Concatenate(axis=-1, name='combine_imgs_d')([img_S, img_T_cropped])
        combined_imgs = Add(name='combine_imgs_d')([img_S, img_T_cropped])
        d1 = d_layer(combined_imgs, self.df, bn=False, name='d1')               # 60
        d2 = d_layer(d1, self.df*2, name='d2')                                  # 60
        pool = MaxPooling3D(pool_size=(2, 2, 2), name='d2_pool')(d2)            # 30

        d3 = d_layer(pool, self.df*4, name='d3')                                # 30
        d4 = d_layer(d3, self.df*8, name='d4')                                  # 30
        pool = MaxPooling3D(pool_size=(2, 2, 2), name='d4_pool')(d4)            # 15

        d5 = d_layer(pool, self.df*8, name='d5')                                # 15

        # ToDo: Use FC layer at the end like specified in the paper
        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid', name='validity')(d5) #9
        #d6 = Conv3D(1, kernel_size=4, strides=1, padding='same', name='validity')(d5)  # 6x6x6

        #validity = Flatten(data_format='channels_last')(d6)
        #x = Reshape((6*6*6*512,))(d5) # hack to avoid flatten bug
        #validity = Dense(1, activation='sigmoid')(x)

        # Use FC layer
        #d6 = Flatten(input_shape=(self.batch_sz,) + (6,6,6,512))(d5)
        #validity = Dense(1, activation='sigmoid')(d5)

        return Model([img_S, img_T], validity, name='discriminator_model')


    """
    Discriminator Network v2
    """
    def build_discriminator_v2(self):

        def d_layer(layer_input, filters, f_size=5, bn=True, name=''):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same', name=name+'_conv3d')(layer_input)
            d = LeakyReLU(alpha=0.2, name=name+'_leakyrelu')(d)
            if bn:
                d = BatchNormalization(momentum=0.8, name=name+'_bn')(d)
            return d

        img_A =  Input(shape=self.input_shape_d, name='input_img_A')            # 68x68x68 warped_img or reference
        img_T = Input(shape=self.input_shape_g, name='input_img_T')             # 148x148x148 template

        img_T_cropped = Cropping3D(cropping=40)(img_T)  # 68x68x68

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_T_cropped])

        d1 = d_layer(combined_imgs, self.df, bn=False, name='d1')
        d2 = d_layer(d1, self.df*2, name='d2')
        d3 = d_layer(d2, self.df*4, name='d3')
        d4 = d_layer(d3, self.df*8, name='d4')

        #d5 = d_layer(d4, self.df*16, name='d5')
       # d5 = Flatten()(d4)
       # validity = Dense(1, activation='sigmoid')(d5)

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid', name='disc_sig')(d4) # 5x5x5

        return Model([img_A, img_T], validity, name='discriminator_model')



    """
    Deformable Transformation Layer    
    """
    def build_transformation(self):
        img_S = Input(shape=self.input_shape_g, name='input_img_S_transform')  # 148
        phi = Input(shape=self.output_shape_g, name='input_phi_transform')     # 60

        img_S_cropped = Cropping3D(cropping=44)(img_S)  # 60
        warped_S = Lambda(dense_image_warp_3D, output_shape=(60,60,60,1))([img_S_cropped, phi])

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
       # return gradients_sqr_sum + lr


    """
    Training
    """
    def train(self, epochs, batch_size=1, sample_interval=50):

        # Adversarial loss ground truths
        disc_patch = self.output_shape_d
        input_sz = 148
        output_sz = 60
        gap = int((input_sz - output_sz)/2)

        # hard labels
        validhard = np.ones((self.batch_sz,) + disc_patch)
        fakehard = np.zeros((self.batch_sz,) + disc_patch)
        # hard labels with only one output
        #validhard = np.ones((self.batch_sz, 1))
        #fakehard = np.zeros((self.batch_sz, 1))

        # soft labels only smooth the labels of postive samples
        # https://arxiv.org/abs/1701.00160
        # https://github.com/soumith/ganhacks/issues/41
        #smooth = 0.1 # validhard -smooth
        #validsoft =  0.9 + 0.1 * np.random.random_sample((self.batch_sz,) + disc_patch)     # random between [0.9, 1)
        #fakesoft =  0.1 * np.random.random_sample((self.batch_sz,) + disc_patch)           # random between [0, 0.1)
        validsoft = np.random.uniform(low=0.7, high=1.2, size=(self.batch_sz,) + disc_patch)
        fakesoft = np.random.uniform(low=0.0, high=0.3, size=(self.batch_sz,) + disc_patch)


        start_time = datetime.datetime.now()
        for epoch in range(epochs):
            for batch_i, (batch_img, batch_img_template, batch_img_golden) in enumerate(self.data_loader.load_batch()):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                #assert not np.any(np.isnan(batch_img))
                #assert not np.any(np.isnan(batch_img_template))
                phi = self.generator.predict([batch_img, batch_img_template]) #24x24x24
                #assert not np.any(np.isnan(phi))

                # deformable transformation
                transform = self.transformation.predict([batch_img, phi])     #24x24x24
                #assert not np.any(np.isnan(transform))

                # Create a ref image by perturbing th subject image with the template image
                perturbation_factor_alpha = 0.1 if epoch > epochs/2 else 0.2
                batch_ref = perturbation_factor_alpha * batch_img + (1- perturbation_factor_alpha) * batch_img_template #64x64x64

                batch_img_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_img.dtype)
                batch_ref_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_ref.dtype)
                batch_temp_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_img_template.dtype)
                #batch_golden_sub = np.zeros((self.batch_sz, output_sz, output_sz, output_sz, self.channels), dtype=batch_img_golden.dtype)

                # take only (24,24,24) from the (64,64,64) size
                batch_img_sub[:, :, :, :, :] = batch_img[:, 0 + gap:0 + gap + output_sz,
                                                            0 + gap:0 + gap + output_sz,
                                                            0 + gap:0 + gap + output_sz, :]
                batch_ref_sub[:, :, :, :, :] = batch_ref[:, 0 + gap:0 + gap + output_sz,
                                                             0 + gap:0 + gap + output_sz,
                                                             0 + gap:0 + gap + output_sz, :]

                #batch_golden_sub[:, :, :, :, :] = batch_img_golden[:, 0 + gap:0 + gap + output_sz,
                #                                            0 + gap:0 + gap + output_sz,
                #                                            0 + gap:0 + gap + output_sz, :]

                batch_temp_sub[:, :, :, :, :] = batch_img_template[:, 0 + gap:0 + gap + output_sz,
                                                                      0 + gap:0 + gap + output_sz,
                                                                      0 + gap:0 + gap + output_sz, :]

                #assert not np.any(np.isnan(batch_img_sub))
                #assert not np.any(np.isnan(batch_ref_sub))
                #assert not np.any(np.isnan(batch_temp_sub))

                # Train the discriminator (R -> T is valid, S -> T is fake)
                # Noisy and soft labels
                noisy_prob = 1 - np.sqrt(1 - np.random.random()) # peak near low values and falling off towards high values
                if noisy_prob < 0.85: # occasionally flip labels to introduce noisy labels
                    d_loss_real = self.discriminator.train_on_batch([batch_ref_sub, batch_img_template], validhard)
                    d_loss_fake = self.discriminator.train_on_batch([transform, batch_img_template], fakehard)
                else:
                    d_loss_real = self.discriminator.train_on_batch([batch_ref_sub, batch_img_template], fakehard)
                    d_loss_fake = self.discriminator.train_on_batch([transform, batch_img_template], validhard)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                # Train the generator (to fool the discriminator)
                g_loss = self.combined.train_on_batch([batch_img, batch_img_template], validhard)

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
                    #self.write_log(self.callback, ['g_loss'], [g_loss[0]], batch_i)
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
        os.makedirs(path+'generated_v148/' , exist_ok=True)

        idx, imgs_S = self.data_loader.load_data(is_validation=True)
        imgs_T = self.data_loader.img_template

        # Mask the image with the template mask # already done in the preparing phase
        # imgs_T_mask = self.data_loader.mask_template
        # imgs_S = imgs_S * imgs_T_mask

        predict_img = np.zeros(imgs_S.shape, dtype=imgs_S.dtype)
        predict_phi = np.zeros(imgs_S.shape + (3,), dtype=imgs_S.dtype)

        input_sz =self.crop_size_g
        step = (60, 60, 60)

        gap = (int((input_sz[0] - step[0]) / 2), int((input_sz[1] - step[1]) / 2), int((input_sz[2] - step[2]) / 2))
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

                    predict_img[row + gap[0]:row + gap[0] + step[0],
                                col + gap[1]:col + gap[1] + step[1],
                                vol + gap[2]:vol + gap[2] + step[2]] = patch_predict_warped[0, :, :, :, 0]
                    predict_phi[row + gap[0]:row + gap[0] + step[0],
                               col + gap[1]:col + gap[1] + step[1],
                               vol + gap[2]:vol + gap[2] + step[2],:] = patch_predict_phi[0, :, :, :, :]


        elapsed_time = datetime.datetime.now() - start_time
        print(" --- Prediction time: %s" % (elapsed_time))

        nrrd.write(path+"generated_v148/%d_%d_%d" % (epoch, batch_i, idx), predict_img)
        self.data_loader._write_nifti(path+"generated_v148/phi%d_%d_%d" % (epoch, batch_i, idx), predict_phi)

        file_name = 'gan_network'
        # save the whole network
        gan.combined.save(path+ 'generated_v148/'+ file_name +str(epoch)+ '.whole.h5', overwrite=True)
        print('Save the whole network to disk as a .whole.h5 file')
        model_jason = gan.combined.to_json()
        with open(path+ 'generated_v148/'+file_name +str(epoch)+ '_arch.json', 'w') as json_file:
            json_file.write(model_jason)
        gan.combined.save_weights(path+ 'generated_v148/'+file_name +str(epoch)+ '_weights.h5', overwrite=True)
        print('Save the network architecture in .json file and weights in .h5 file')

        # # save the generator network
        # gan.generator.save(path+ 'generated_v148/'+file_name +str(epoch)+ '.gen.h5', overwrite=True)
        # print('Save the generator network to disk as a .whole.h5 file')
        # model_jason = gan.generator.to_json()
        # with open(path+ 'generated_v148/'+file_name +str(epoch)+ '_gen_arch.json', 'w') as json_file:
        #     json_file.write(model_jason)
        # gan.generator.save_weights(path+ 'generated_v148/'+file_name +str(epoch)+ '_gen_weights.h5', overwrite=True)
        # print('Save the generator architecture in .json file and weights in .h5 file')


if __name__ == '__main__':
    # Use GPU
    K.tensorflow_backend._get_available_gpus()
    K.image_dim_ordering()
    K.set_image_dim_ordering('tf')
    # launch tf debugger
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)

    gan = GANUnetModel148()
    gan.train(epochs=20000, batch_size=4, sample_interval=200)





