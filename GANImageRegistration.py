from __future__ import print_function, division

#from tensorflow.contrib.image import dense_image_warp

import keras.backend as K
from keras.datasets import mnist

from keras.layers import BatchNormalization, Activation, MaxPooling3D, Cropping3D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate
from keras.layers import Lambda

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling3D, Conv3D

from keras.optimizers import RMSprop, Adam

from keras.models import Model
from keras.utils import to_categorical

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import scipy
import sys



class GANImageRegistration:
    def __init__(self):
        self.crop_size_g = (64, 64, 64)
        self.crop_size_d = (24, 24, 24)
        self.channels = 1
        self.input_shape_g = self.crop_size_g + (self.channels,)
        self.input_shape_d = self.crop_size_d + (self.channels,)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])


        # Build the generator
        self.generator = self.build_generator()

        # Input images
        img_S = Input(shape=self.input_shape_g) # subject image S
        img_T = Input(shape=self.input_shape_g) # template image T
        img_R = Input(shape=self.input_shape_d) # reference image R

        # By conditioning on T generate a warped transformation of S
        phi = self.generator([img_S, img_T])

        # warp the subject image # TODO: maybe put this in the generator network to evaluate the gradient on it as well
        img_S_downsampled = Cropping3D(cropping=4)(MaxPooling3D(pool_size=(2,2,2))(img_S)) # 24x24x24
        # Put the warping function in a Lambda layer because it uses tensorflow
        warped_S = Lambda(self.dense_image_warp_3D)(img_S_downsampled, phi)

        # Use Python partial to provide loss function with additional deformable field argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          phi=phi)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([warped_S, img_R])

        self.combined = Model(inputs=[img_S, img_T], outputs=valid)
        self.combined.compile(loss=self.partial_gp_loss,
                              #loss_weights=[1, 100],
                              optimizer=optimizer)


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
        inputs_S = Input(shape=input_shape)  # 64x64x64
        inputs_T = Input(shape=input_shape)  # 64x64x64

        # Concatenate subject image and template image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([inputs_S, inputs_T])

        # down-sampling
        down1 = g_layer(input_tensor=combined_imgs, n_filters=self.gf, padding='valid')  # 62x62x62
        down1 = g_layer(input_tensor=down1, n_filters=self.gf, padding='valid')  # 60x60x60
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(down1)  # 30x30x30

        down2 = g_layer(input_tensor=pool1, n_filters=2 * self.gf, padding='valid')  # 28x28x28
        down2 = g_layer(input_tensor=down2, n_filters=2 * self.gf, padding='valid')  # 26x26x26
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(down2)  # 13x13x13

        center = g_layer(input_tensor=pool2, n_filters=4 * self.gf, padding='valid')  # 11x11x11
        center = g_layer(input_tensor=center, n_filters=4 * self.gf, padding='valid')  # 9x9x9

        # up-sampling
        up2 = concatenate(
            [Cropping3D(((4, 4), (4, 4), (4, 4)))(down2), UpSampling3D(size=(2, 2, 2))(center)])  # 18x18x18
        up2 = g_layer(input_tensor=up2, n_filters=2 * self.gf, padding='valid')  # 16x16x16
        up2 = g_layer(input_tensor=up2, n_filters=2 * self.gf, padding='valid')  # 14x14x14

        up1 = concatenate(
            [Cropping3D(((16, 16), (16, 16), (16, 16)))(down1), UpSampling3D(size=(2, 2, 2))(up2)])  # 28x28x28
        up1 = g_layer(input_tensor=up1, n_filters=self.gf, padding='valid')  # 26x26x26
        up1 = g_layer(input_tensor=up1, n_filters=self.gf, padding='valid')  # 24x24x24

        # ToDo: check if the activation function 'sigmoid' is the right one or leave it to be linear; originally sigmoid
        phi = Conv3D(filters=1, kernel_size=(1, 1, 1), use_bias=False)(up1)  # 24x24x24

        model = Model(inputs=combined_imgs, outputs=phi)

        return model

    """
    Descriminator Network
    """
    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=3, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = ReLU()(d)
            if bn:
                d = BatchNormalization()(d)
            return d

        img_A = Input(shape=self.input_shape_d) #24x24x24
        img_B = Input(shape=self.input_shape_d) #24x24x24

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False) #24x24x24
        d2 = d_layer(d1, self.df*2)                    #24x24x24
        pool = MaxPooling3D(pool_size=(2, 2, 2))(d2)   #12x12x12

        d3 = d_layer(pool, self.df*4)                  #12x12x12
        d4 = d_layer(d3, self.df*8)                    #12x12x12
        pool = MaxPooling3D(pool_size=(2, 2, 2))(d4)   #6x6x6

        d4 = d_layer(pool, self.df*8)                  #6x6x6

        # ToDo: check if the activation function 'sigmoid' is the right one or leave it to be linear; originally linear
        # ToDo: Use FC layer at the end like specified in the paper
        validity = Conv3D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4) #6x6x6
        #validity = Dense(1, activation='sigmoid')(d4)

        return Model([img_A, img_B], validity)


    """
    Deformable transformation layer
    """
    def deformable_transformation_layer(self, deformable_field, subject_image):
        # TODO: trilinear interpolation
        return subject_image


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
        gradients = self.numerical_gradient_3D(phi)

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
    Define numerical gradient
    """
    def numerical_gradient_3D(self, phi):
        """ Calculate the numerical gradient of a tensor
        similar to Matlab gradient(F)
        This implementation only considers neighbors in the same dim
        TODO: take the gradient across the 8 neighbors of the voxel
        Args:
            phi: 5D tensor (b, j, i, k, c)
        Return:
            G: numerical gradient -  same dimension as phi: 5D tensor
        """
        _, Nx, Ny, Nz, _ = phi.shape.as_list()
        # ... calculates the central difference for interior data points --> like Matlab: gradient(F)
        # G(:,j) = 0.5*(A(:,j+1) - A(:,j-1));
        Gy_c = 0.5 * (phi[:, 2:, :, :, :] - phi[:, :-2, :, :, :])  # (b, 22, 24, 24, 1)
        Gx_c = 0.5 * (phi[:, :, 2:, :, :] - phi[:, :, :-2, :, :])  # (b, 22 23, 24, 1)
        Gz_c = 0.5 * (phi[:, :, :, 2:, :] - phi[:, :, :, :-2, :])  # (b, 22, 24, 23, 1)

        # ... calculate values along the edges of the matrix with single-sides differences
        Gy_N = phi[:, Nx - 1:Nx, :, :, :] - phi[:, Nx - 2:Nx - 1, :, :, :]
        Gy_0 = phi[:, 1:2, :, :, :] - phi[:, 0:1, :, :, :]

        Gx_N = phi[:, :, Ny - 1:Ny, :, :] - phi[:, :, Ny - 2:Ny - 1, :, :]
        Gx_0 = phi[:, :, 1:2, :, :] - phi[:, :, 0:1, :, :]

        Gz_N = phi[:, :, :, Nz - 1:Nz, :] - phi[:, :, :, Nz - 2:Nz - 1, :]
        Gz_0 = phi[:, :, :, 1:2, :] - phi[:, :, :, 0:1, :]

        # ... concatenate the edge differences with the central differences
        Gy = K.concatenate([Gy_0, Gy_c, Gy_N], axis=1)
        Gx = K.concatenate([Gx_0, Gx_c, Gx_N], axis=2)
        Gz = K.concatenate([Gz_0, Gz_c, Gz_N], axis=3)

        # ... then add the partial gradients
        G = Gx + Gy + Gz

        return G

    """
    Define image warping 
    """
    # This use tf and will be wrapped to be used in Lambda layer in Keras
    def dense_image_warp_3D(self, image, flow, name='dense_image_warp'):
        """Image warping using per-pixel flow vectors.

        Apply a non-linear warp to the image, where the warp is specified by a dense
        flow field of offset vectors that define the correspondences of pixel values
        in the output image back to locations in the  source image. Specifically, the
        pixel value at output[b, j, i, k, c] is
        images[b, j - flow[b, j, i, k, 0], i - flow[b, j, i, k, 1], k - flow[b, j, i, k, 2], c].
        The locations specified by this formula do not necessarily map to an int
        index. Therefore, the pixel value is obtained by trilinear
        interpolation of the 8 nearest pixels around
        (b, j - flow[b, j, i, k, 0], i - flow[b, j, i, k, 1], k - flow[b, j, i, k, 2]). For locations outside
        of the image, we use the nearest pixel values at the image boundary.
        Args:
          image: 5-D float `Tensor` with shape `[batch, height, width, depth, channels]`.
          flow: A 5-D float `Tensor` with shape `[batch, height, width, depth, 3]`.
          name: A name for the operation (optional).
          Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
          and do not necessarily have to be the same type.
        Returns:
          A 5-D float `Tensor` with shape`[batch, height, width, depth, channels]`
            and same type as input image.
        Raises:
          ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                      of dimensions.
        """

        batch_size, height, width, depth, channels = (array_ops.shape(image)[0],
                                                    array_ops.shape(image)[1],
                                                    array_ops.shape(image)[2],
                                                    array_ops.shape(image)[3],
                                                    array_ops.shape(image)[4])

        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y, grid_z = array_ops.meshgrid(
            math_ops.range(width), math_ops.range(height), math_ops.range(depth))
        stacked_grid = math_ops.cast(
            array_ops.stack([grid_y, grid_x, grid_z], axis=3), flow.dtype)
        batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = array_ops.reshape(query_points_on_grid,
                                                   [batch_size, height * width * depth, 3])
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = self.interpolate_trilinear(image, query_points_flattened)
        interpolated = array_ops.reshape(interpolated,
                                         [batch_size, height, width, depth, channels])
        return interpolated



    def interpolate_trilinear(self, grid, query_points, name='interpolate_trilinear', indexing='ijk'):
        """Similar to Matlab's interp2 function but for 3D.
        Finds values for query points on a grid using trilinear interpolation.
        Args:
            grid: a 5-D float `Tensor` of shape `[batch, height, width, depth, channels]`.
            query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 3]`.
            name: a name for the operation (optional).
            indexing: whether the query points are specified as row and column (ijk),
                or Cartesian coordinates (xyz).
        Returns:
            values: a 3-D `Tensor` with shape `[batch, N, channels]`
        Raises:
            ValueError: if the indexing mode is invalid, or if the shape of the inputs
                invalid.
        """

        if indexing != 'ijk' and indexing != 'xyz':
            raise ValueError('Indexing mode must be \'ijk\' or \'xyz\'')

        with ops.name_scope(name):
            grid = ops.convert_to_tensor(grid)
            query_points = ops.convert_to_tensor(query_points)
            shape = array_ops.unstack(array_ops.shape(grid))
            if len(shape) != 5:
                msg = 'Grid must be 5 dimensional. Received: '
                raise ValueError(msg + str(shape))

            batch_size, height, width, depth, channels = shape
            grid_type = grid.dtype

            query_type = query_points.dtype
            query_shape = array_ops.unstack(array_ops.shape(query_points))


            if len(query_shape) != 3:
                msg = ('Query points must be 3 dimensional. Received: ')
                raise ValueError(msg + str(query_shape))

            _, num_queries, _ = query_shape

            alphas = []
            floors = []
            ceils = []

            index_order = [0, 1, 2] if indexing == 'ijk' else [2, 1, 0]
            unstacked_query_points = array_ops.unstack(query_points, axis=2)

            for dim in index_order:
                with ops.name_scope('dim-' + str(dim)):
                    queries = unstacked_query_points[dim]

                    size_in_indexing_dimension = shape[dim + 1] # because batch size is the first index in shape

                    # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                    # is still a valid index into the grid.
                    max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                    min_floor = constant_op.constant(0.0, dtype=query_type)
                    floor = math_ops.minimum(
                        math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                    int_floor = math_ops.cast(floor, dtypes.int32)
                    floors.append(int_floor)
                    ceil = int_floor + 1
                    ceils.append(ceil)

                    # alpha has the same type as the grid, as we will directly use alpha
                    # when taking linear combinations of pixel values from the image.
                    alpha = math_ops.cast(queries - floor, grid_type)
                    min_alpha = constant_op.constant(0.0, dtype=grid_type)
                    max_alpha = constant_op.constant(1.0, dtype=grid_type)
                    alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                    # Expand alpha to [b, n, 1] so we can use broadcasting
                    # (since the alpha values don't depend on the channel).
                    alpha = array_ops.expand_dims(alpha, 2)
                    alphas.append(alpha)

            flattened_grid = array_ops.reshape(grid,
                                               [batch_size * height * width * depth, channels])
            batch_offsets = array_ops.reshape(
                math_ops.range(batch_size) * height * width * depth, [batch_size, 1])

            # This wraps array_ops.gather. We reshape the image data such that the
            # batch, y, x and z coordinates are pulled into the first dimension.
            # Then we gather. Finally, we reshape the output back. It's possible this
            # code would be made simpler by using array_ops.gather_nd.
            def gather(y_coords, x_coords, z_coords, name):
                with ops.name_scope('gather-' + name):
                    # map a 3d coordinates to a single number
                    linear_coordinates = batch_offsets + y_coords * width + x_coords * height + z_coords * depth
                    gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                    return array_ops.reshape(gathered_values,
                                             [batch_size, num_queries, channels])

            # grab the pixel values in the 8 corners around each query point
            # coordinates: y x z (rows(height), columns(width), depth)      # yxz
            fy_fx_fz = gather(floors[0], floors[1], floors[2], 'fy_fx_fz')  # 000
            fy_fx_cz = gather(floors[0], floors[1], ceils[2], 'fy_fx_cz')   # 001
            fy_cx_fz = gather(floors[0], ceils[1], floors[2], 'fy_cx_fz')   # 010
            fy_cx_cz = gather(floors[0], ceils[1], ceils[2], 'fy_cx_cz')    # 011
            cy_fx_fz = gather(ceils[0], floors[1], floors[2], 'cy_fx_fz')   # 100
            cy_fx_cz = gather(ceils[0], floors[1], ceils[2], 'cy_fx_cz')    # 101
            cy_cx_fz = gather(ceils[0], ceils[1], floors[2], 'cy_cx_fz')    # 110
            cy_cx_cz = gather(ceils[0], ceils[1], ceils[2], 'cy_cx_cz')     # 111

            # now, do the actual interpolation
            with ops.name_scope('interpolate'):
                # we perform 4 linear interpolation to compute a, b, c, and d using alpha[1],
                # then we compute e and f by interpolating a, b, c and d  using alpha[0],
                # and finally we find our sample point by interpolating e and f using alpha[2]
                a = alpha[1] * (cy_cx_fz - cy_fx_fz) + cy_fx_fz
                b = alpha[1] * (fy_cx_fz - fy_fx_fz) + fy_fx_fz
                c = alpha[1] * (cy_cx_cz - cy_fx_cz) + cy_fx_cz
                d = alpha[1] * (fy_cx_cz - fy_fx_cz) + fy_fx_cz
                e = alpha[0] * (b - a) + a
                f = alpha[0] * (d - c) + c
                g = alpha[2] * (f - e) + e

            return g