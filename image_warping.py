import numpy as np


def dense_image_warp_3D_dipy(image, flow):
    DEBUG = 0

    batch_size, height, width, depth, channels = (np.shape(image)[0],
                                                np.shape(image)[1],
                                                np.shape(image)[2],
                                                np.shape(image)[3],
                                                np.shape(image)[4])
    input_sz = 64
    output_sz = 24
    gap = int((input_sz - output_sz) / 2)
    batch_img_sub = np.zeros((batch_size, output_sz, output_sz, output_sz, channels), dtype=image.dtype)
    batch_img_sub[:, :, :, :, :] = image[:, 0 + gap:0 + gap + output_sz,
                                   0 + gap:0 + gap + output_sz,
                                   0 + gap:0 + gap + output_sz, :]


    warped_img = np.zeros(shape=np.shape(batch_img_sub) , dtype=image.dtype)
    image_perm = np.transpose(image, axes=[0, 3, 1, 2, 4]) # b, k, i, j, c
    flow_perm = np.transpose(flow, axes=[0, 3, 1, 2, 4] )  # b, k, i, j, 3

    for b in range(batch_size):
        volume = image_perm[b, :,:,:, 0]
        warped_img[b, :,:,:, 0] = _warp_3d(volume, flow_perm[b,:,:,:,:]) # (b,k,i,j,c)

    warped_img = np.transpose(warped_img, axes=[0, 2, 3, 1, 4])  #(b,i,j,k,c)

    return warped_img



def _warp_3d(volume, d1):

    """ Warps a 3D volume using trilinear interpolation
    Deforms the input volume under the given transformation. The warped volume
    is computed using tri-linear interpolation and is given by:
    (1) warped[i] = volume[ C * d1[A*i] + B*i ]
    where A = affine_idx_in, B = affine_idx_out, C = affine_disp and i denotes
    the discrete coordinates of a voxel in the sampling grid of
    shape = out_shape.
    Parameters
    ----------
    volume : float array, shape (S, R, C)
        the input volume to be transformed
    d1 : float array, shape (S', R', C', 3)
        the displacement field driving the transformation
    affine_idx_in : double array, shape (4, 4)
        the matrix A in eq. (1) above
    affine_idx_out : double array, shape (4, 4)
        the matrix B in eq. (1) above
    affine_disp :  double array, shape (4, 4)
        the matrix C in eq. (1) above
    out_shape : array, shape (3,)
        the number of slices, rows and columns of the sampling grid
    Returns
    -------
    warped : array, shape = out_shape
        the transformed volume
    Notes
    -----
    To illustrate the use of this function, consider a displacement field d1
    with grid-to-space transformation R, a volume with grid-to-space
    transformation T and let's say we want to sample the warped volume on a
    grid with grid-to-space transformation S (sampling grid). For each voxel
    in the sampling grid with discrete coordinates i, the warped volume is
    given by:
    (2) warped[i] = volume[Tinv * ( d1[Rinv * S * i] + S * i ) ]
    where Tinv = T^{-1} and Rinv = R^{-1}. By identifying A = Rinv * S,
    B = Tinv * S, C = Tinv we can use this function to efficiently warp the
    input image.
    """

    nslices = np.shape(d1)[0]
    nrows =  np.shape(d1)[1]
    ncols = np.shape(d1)[2]

    warped = np.zeros(shape=(nslices, nrows, ncols), dtype=volume.dtype)

    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                dkk = d1[k, i, j, 0]
                dii = d1[k, i, j, 1]
                djj = d1[k, i, j, 2]

                dk = dkk
                di = dii
                dj = djj

                dkk = dk + k
                dii = di + i
                djj = dj + j

                warped[k,i,j] = _interpolate_scalar_3d(volume, dkk, dii, djj)

    return warped

# from dipy
#https://github.com/nipy/dipy/blob/master/dipy/align/vector_fields.pyx
def _interpolate_scalar_3d(volume, dkk, dii, djj):
    """ Trilinear interpolation of a 3D scalar image
    Interpolates the 3D image at (dkk, dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the image's domain,
    zero is written to out instead.
    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position
    Returns
    -------
    out: the value which the interpolation result will be written to
    inside : int
        if (dkk, dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    ns = volume.shape[0]
    nr = volume.shape[1]
    nc = volume.shape[2]

    if not (-1 < dkk < ns and -1 < dii < nr and -1 < djj < nc):
        out = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = np.floor(dkk).astype('int')
    ii = np.floor(dii).astype('int')
    jj = np.floor(djj).astype('int')
    # no one is affected
    cgamma = (dkk - kk).astype('float32')
    calpha = (dii - ii).astype('float32')
    cbeta = (djj - jj).astype('float32')
    alpha = (1.0 - calpha).astype('float32')
    beta = (1.0 - cbeta).astype('float32')
    gamma = (1.0 - cgamma).astype('float32')

    inside = 0
    # ---top-left
    if (ii >= 0) and (jj >= 0) and (kk >= 0):
        out = alpha * beta * gamma * volume[kk, ii, jj]
        inside += 1
    else:
        out = 0
    # ---top-right
    jj += 1
    if (ii >= 0) and (jj < nc) and (kk >= 0):
        out += alpha * cbeta * gamma * volume[kk, ii, jj]
        inside += 1
    # ---bottom-right
    ii += 1
    if (ii < nr) and (jj < nc) and (kk >= 0):
        out += calpha * cbeta * gamma * volume[kk, ii, jj]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (ii < nr) and (jj >= 0) and (kk >= 0):
        out += calpha * beta * gamma * volume[kk, ii, jj]
        inside += 1
    kk += 1
    if(kk < ns):
        ii -= 1
        if (ii >= 0) and (jj >= 0):
            out += alpha * beta * cgamma * volume[kk, ii, jj]
            inside += 1
        jj += 1
        if (ii >= 0) and (jj < nc):
            out += alpha * cbeta * cgamma * volume[kk, ii, jj]
            inside += 1
        # ---bottom-right
        ii += 1
        if (ii < nr) and (jj < nc):
            out += calpha * cbeta * cgamma * volume[kk, ii, jj]
            inside += 1
        # ---bottom-left
        jj -= 1
        if (ii < nr) and (jj >= 0):
            out += calpha * beta * cgamma * volume[kk, ii, jj]
            inside += 1

    # assert that inside == 8
    #return 1 if inside == 8 else 0
    return out
