from __future__ import print_function, division

import keras.backend as K

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops


__author__ = 'elmalakis'

"""
Define trilinear interpolation
It uses tensorflow array operations so this function has to be wrapped in a lambda layer before being used in keras
"""
def interpolate_trilinear(grid, query_points, name='interpolate_trilinear', indexing='ijk'):
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
    DEBUG = 0
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

        index_order = [0, 1, 2] if indexing == 'ijk' else [1, 0, 2]
        unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in index_order:
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]
                size_in_indexing_dimension = shape[dim + 1]  # because batch size is the first index in shape

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                if DEBUG: int_floor = K.print_tensor(int_floor, message='int_floor is:')
                floors.append(int_floor)
                ceil = int_floor + 1
                if DEBUG: ceil = K.print_tensor(ceil, message='ceil is:')
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
                if DEBUG: alpha = K.print_tensor(alpha, message='alpha is:')
                alphas.append(alpha)
                K.print_tensor(alpha)

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
                # https://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
                linear_coordinates = batch_offsets + x_coords + y_coords * width + z_coords * (width*height)
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 8 corners around each query point
        # coordinates: y x z (rows(height), columns(width), depth)      # yxz
        fy_fx_fz = gather(floors[0], floors[1], floors[2], 'fy_fx_fz')  # 000
        fy_fx_cz = gather(floors[0], floors[1], ceils[2], 'fy_fx_cz')  # 001
        fy_cx_fz = gather(floors[0], ceils[1], floors[2], 'fy_cx_fz')  # 010
        fy_cx_cz = gather(floors[0], ceils[1], ceils[2], 'fy_cx_cz')  # 011
        cy_fx_fz = gather(ceils[0], floors[1], floors[2], 'cy_fx_fz')  # 100
        cy_fx_cz = gather(ceils[0], floors[1], ceils[2], 'cy_fx_cz')  # 101
        cy_cx_fz = gather(ceils[0], ceils[1], floors[2], 'cy_cx_fz')  # 110
        cy_cx_cz = gather(ceils[0], ceils[1], ceils[2], 'cy_cx_cz')  # 111

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            # we perform 4 linear interpolation to compute a, b, c, and d using alpha[1],
            # then we compute e and f by interpolating a, b, c and d  using alpha[0],
            # and finally we find our sample point by interpolating e and f using alpha[2]
            with ops.name_scope('alpha-a'): a = alphas[1] * (cy_cx_fz - cy_fx_fz) + cy_fx_fz
            with ops.name_scope('alpha-b'): b = alphas[1] * (fy_cx_fz - fy_fx_fz) + fy_fx_fz
            with ops.name_scope('alpha-c'): c = alphas[1] * (cy_cx_cz - cy_fx_cz) + cy_fx_cz
            with ops.name_scope('alpha-d'): d = alphas[1] * (fy_cx_cz - fy_fx_cz) + fy_fx_cz
            with ops.name_scope('alpha-e'): e = alphas[0] * (b - a) + a
            with ops.name_scope('alpha-f'): f = alphas[0] * (d - c) + c
            with ops.name_scope('alpha-g'): g = alphas[2] * (f - e) + e

        return g


"""
Define numerical gradient
Use K backend so it can be used directly without wrapping it in a lambda layer
"""
def numerical_gradient_3D(phi):
    """ Calculate the numerical gradient of a tensor
    similar to Matlab gradient(F)
    This implementation only considers neighbors in the same dim
    TODO: take the gradient across the 8 neighbors of the voxel
    Args:
        phi: 5-D float `Tensor` with shape `[batch, height, width, depth, channels]`.
    Return:
        G: A 5-D float `Tensor` with shape`[batch, height, width, depth, channels]`
        and same type as input phi.
    """
    DEBUG = 0
    _, Nx, Ny, Nz, _ = phi.shape.as_list()
    if DEBUG: phi = K.print_tensor(phi, message='phi is:')
    # ... calculates the central difference for interior data points --> like Matlab: gradient(F)
    # G(:,j) = 0.5*(A(:,j+1) - A(:,j-1));
    Gy_c = 0.5 * (phi[:, 2:, :, :, :] - phi[:, :-2, :, :, :])  # (b, 22, 24, 24, 1)
    Gx_c = 0.5 * (phi[:, :, 2:, :, :] - phi[:, :, :-2, :, :])  # (b, 22 23, 24, 1)
    Gz_c = 0.5 * (phi[:, :, :, 2:, :] - phi[:, :, :, :-2, :])  # (b, 22, 24, 23, 1)
    if DEBUG: Gy_c = K.print_tensor(Gy_c, message='Gy_c is:')
    if DEBUG: Gx_c = K.print_tensor(Gx_c, message='Gx_c is:')
    if DEBUG: Gz_c = K.print_tensor(Gz_c, message='Gz_c is:')
    # ... calculate values along the edges of the matrix with single-sides differences
    Gy_N = phi[:, Nx - 1:Nx, :, :, :] - phi[:, Nx - 2:Nx - 1, :, :, :]
    Gy_0 = phi[:, 1:2, :, :, :] - phi[:, 0:1, :, :, :]
    if DEBUG: Gy_N = K.print_tensor(Gy_N, message='Gy_N is:')
    if DEBUG: Gy_0 = K.print_tensor(Gy_0, message='Gy_0 is:')

    Gx_N = phi[:, :, Ny - 1:Ny, :, :] - phi[:, :, Ny - 2:Ny - 1, :, :]
    Gx_0 = phi[:, :, 1:2, :, :] - phi[:, :, 0:1, :, :]
    if DEBUG: Gx_N = K.print_tensor(Gx_N, message='Gx_N is:')
    if DEBUG: Gx_0 = K.print_tensor(Gx_0, message='Gx_0 is:')

    Gz_N = phi[:, :, :, Nz - 1:Nz, :] - phi[:, :, :, Nz - 2:Nz - 1, :]
    Gz_0 = phi[:, :, :, 1:2, :] - phi[:, :, :, 0:1, :]
    if DEBUG: Gz_N = K.print_tensor(Gz_N, message='Gx_N is:')
    if DEBUG: Gz_0 = K.print_tensor(Gz_0, message='Gx_0 is:')

    # ... concatenate the edge differences with the central differences
    Gy = K.concatenate([Gy_0, Gy_c, Gy_N], axis=1)
    Gx = K.concatenate([Gx_0, Gx_c, Gx_N], axis=2)
    Gz = K.concatenate([Gz_0, Gz_c, Gz_N], axis=3)
    if DEBUG: Gy = K.print_tensor(Gy, message='Gy is:')
    if DEBUG: Gx = K.print_tensor(Gx, message='Gx is:')
    if DEBUG: Gz = K.print_tensor(Gz, message='Gz is:')

    # ... then add the partial gradients
    G = Gx + Gy + Gz

    return G


"""
Define image warping 
"""
# This use tf and will be wrapped to be used in Lambda layer in Keras
def dense_image_warp_3D(tensors, name='dense_image_warp'):
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the source image. Specifically, the
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

    DEBUG = 0
    image = tensors[0]
    flow = tensors[1]

    batch_size, height, width, depth, channels = (array_ops.shape(image)[0],
                                                array_ops.shape(image)[1],
                                                array_ops.shape(image)[2],
                                                array_ops.shape(image)[3],
                                                array_ops.shape(image)[4])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y, grid_z = array_ops.meshgrid(math_ops.range(width), math_ops.range(height), math_ops.range(depth))
    stacked_grid = math_ops.cast(array_ops.stack([grid_y, grid_x, grid_z], axis=3), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0) #add the batch dim on axis 0

    query_points_on_grid = batched_grid - flow
    if DEBUG: query_points_on_grid= K.print_tensor(query_points_on_grid, message="query_points_on_grid is:")

    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, height * width * depth, 3])
    if DEBUG: query_points_flattened = K.print_tensor(query_points_flattened, message="query_points_flattened is:")
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = interpolate_trilinear(image, query_points_flattened)
    if DEBUG: interpolated = K.print_tensor(interpolated, message='interpolated is:')
    interpolated = array_ops.reshape(interpolated,
                                     [batch_size, height, width, depth, channels])
    return interpolated