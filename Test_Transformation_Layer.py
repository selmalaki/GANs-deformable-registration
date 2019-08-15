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

from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model

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


class TestTransformationLayer():

    def __init__(self):
        self.batch_sz = 1
        self.channels = 1
        self.crop_size_g = (64, 64, 64)
        self.input_shape_g = self.crop_size_g + (self.channels,)
        self.output_shape_g =  (64, 64, 64)+ (3,)  # phi has three outputs. one for each X, Y, and Z dimensions

        print('--- read sample_image ---')
        self.img = self.read_sample_image()
        print('--- read sample phi ---')
        self.phi = self.read_sample_phi()
        print('--- build transformation layer ---')
        self.transformation = self.build_transformation()
        self.transformation.summary()


    def read_sample_image(self, idx=0):
        filepath = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
        curr_img = filepath + 'proc/' + '20161102_32_C1_Scope_1_C1_down_result.nrrd'
        mask_template = filepath + 'preprocessed_convexhull/JRC2018_lo_dilated_mask.nrrd'

        mask_template, templ_header = nrrd.read(mask_template)
        curr_img, curr_img_header =  nrrd.read(curr_img)

        mask_template = np.float32(mask_template)
        curr_img = np.float32(curr_img)

        mask = 1 - mask_template
        mask = np.uint8(mask)
        img_masked = np.ma.array(curr_img, mask=mask)
        curr_img = (curr_img - np.mean(img_masked)) / np.std(img_masked)

        print('img size: '+str(curr_img.shape))
        return curr_img

    def read_sample_phi(self, idx=0):
        goldenfilepath = '/nrs/saalfeld/john/public/forSalma/lo_res/proc/'
        curr_phi = goldenfilepath + '20161102_32_C1_Scope_1_C1_down/deformationField_noAffine.nrrd'
        curr_phi, phi_header = nrrd.read(curr_phi)
        print('phi size: '+str(curr_phi.shape)) #(3,1121, 546, 334)
        curr_phi = curr_phi.transpose(1,2,3,0)
        print('phi size: ' + str(curr_phi.shape))  # (1121, 546, 334, 3)
        return np.float32(curr_phi)



    """
    Deformable Transformation Layer    
    """
    def build_transformation(self):
        img_S = Input(shape=self.input_shape_g, name='input_img_S_transform')  # 64x64x64
        phi = Input(shape=self.output_shape_g, name='input_phi_transform')     # 24x24x24

        #img_S_cropped = Cropping3D(cropping=20)(img_S)  # 24x24x24
        warped_S = Lambda(dense_image_warp_3D, output_shape=(64,64,64,1))([img_S, phi])

        return Model([img_S, phi], warped_S,  name='transformation_layer')


    def sample_images(self, epoch=0, batch_i=0):
        path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
        os.makedirs(path+'generated_with_phi/' , exist_ok=True)

        idx = 0
        predict_img = np.zeros(self.img.shape, dtype=self.img.dtype)

        input_sz = (64, 64, 64)
        step = (24, 24, 24)

        gap = (int((input_sz[0] - step[0]) / 2), int((input_sz[1] - step[1]) / 2), int((input_sz[2] - step[2]) / 2))
        start_time = datetime.datetime.now()
        print('--- start transformation ---')
        for row in range(0, self.img.shape[0] - input_sz[0], step[0]):
            for col in range(0, self.img.shape[1] - input_sz[1], step[1]):
                for vol in range(0, self.img.shape[2] - input_sz[2], step[2]):
                    patch_sub_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=self.img.dtype)
                    patch_sub_phi = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 3), dtype=self.phi.dtype)

                    patch_sub_img[0, :, :, :, 0] = self.img[row:row + input_sz[0],
                                                          col:col + input_sz[1],
                                                          vol:vol + input_sz[2]]
                    patch_sub_phi[0, :, :, :, :] = self.phi[row:row + input_sz[0],
                                                          col:col + input_sz[1],
                                                          vol:vol + input_sz[2]]

                    patch_predict_warped = self.transformation.predict([patch_sub_img, patch_sub_phi])

                    predict_img[row :row + input_sz[0],
                                col :col + input_sz[1],
                                vol :vol + input_sz[2]] = patch_predict_warped[0, :, :, :, 0]

        elapsed_time = datetime.datetime.now() - start_time
        print(" --- Prediction time: %s" % (elapsed_time))

        nrrd.write(path+"generated_with_phi/%d_%d_%d" % (epoch, batch_i, idx), predict_img)



if __name__ == '__main__':
    # Use GPU
    K.tensorflow_backend._get_available_gpus()
    K.image_dim_ordering()
    K.set_image_dim_ordering('tf')
    # launch tf debugger
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)

    transform = TestTransformationLayer()
    transform.sample_images()
