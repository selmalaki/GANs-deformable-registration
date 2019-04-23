
import scipy.ndimage.morphology as morph
import skimage.morphology as skimorph
from skimage import measure
from skimage import filters
from skimage import feature
from scipy import spatial
from scipy import ndimage
import nibabel as nib
import numpy as np
import scipy.misc
import nrrd
import os


__author__ = 'elmalakis'

class PreProcessing():

    def __init__(self):
        pass

    def preprocess(self, path, outdir):
        filesdir = path + 'proc/'
        template = 'JRC2018_lo.nrrd'
        filelist = ['20161102_32_C1_Scope_1_C1_down_result.nrrd',
                    '20161102_32_C3_Scope_4_C1_down_result.nrrd',
                    '20161102_32_D1_Scope_1_C1_down_result.nrrd',
                    '20161102_32_D2_Scope_1_C1_down_result.nrrd',
                    '20161102_32_E1_Scope_1_C1_down_result.nrrd',
                    '20161102_32_E3_Scope_4_C1_down_result.nrrd',
                    '20161220_31_I1_Scope_2_C1_down_result.nrrd',
                    '20161220_31_I2_Scope_6_C1_down_result.nrrd',
                    '20161220_31_I3_Scope_6_C1_down_result.nrrd',
                    '20161220_32_C1_Scope_3_C1_down_result.nrrd',
                    '20161220_32_C3_Scope_3_C1_down_result.nrrd',
                    '20170223_32_A2_Scope_3_C1_down_result.nrrd',
                    '20170223_32_A3_Scope_3_C1_down_result.nrrd',
                    '20170223_32_A6_Scope_2_C1_down_result.nrrd',
                    '20170223_32_E1_Scope_3_C1_down_result.nrrd',
                    '20170223_32_E2_Scope_3_C1_down_result.nrrd',
                    '20170223_32_E3_Scope_3_C1_down_result.nrrd',
                    '20170301_31_B1_Scope_1_C1_down_result.nrrd',
                    '20170301_31_B3_Scope_1_C1_down_result.nrrd',
                    '20170301_31_B5_Scope_1_C1_down_result.nrrd']

        for f in filelist:
            self.create_mask(path=filesdir + f, outdir=outdir)
            print('--- Done ' + f + ' ---')

        self.create_mask(path=path + template, outdir=outdir)


    def create_mask(self, path, outdir):
        """create a mask using otsu thresholding"""
        filename = os.path.basename(path)
        filename = os.path.splitext(filename)[0]

        image, header = nrrd.read(path)
        image = self.normalize_intensity(image)
        image = self.hist_equalization(image)
        nrrd.write(outdir+filename+ '_histogram_normalized.nrrd', image, header=header)

        normalized_image = (image - np.mean(image)) / np.std(image)
        nrrd.write(outdir + filename + '_normalized.nrrd', normalized_image, header=header)

        val = filters.threshold_otsu(image)
        # convert to binary to speed up next processing
        blobs = image > val
        # Dilation enlarges bright regions and shrinks dark regions
        mask = morph.binary_dilation(blobs, iterations=2)
        # Closing remove pepper spots and connects small bright cracks. Close up dark gaps between bright features
        mask = morph.binary_closing(mask, iterations=2)
        mask = morph.binary_fill_holes(mask)

        # Create a convex hull - Not the fastest way
        #for z in range(0, mask.shape[2]):
        #    mask[:,:,z] = skimorph.convex_hull_image(mask[:,:,z])

        #mask = spatial.ConvexHull(mask, incremental = True)
        # convert to 0 and 1 float
        mask = measure.label(mask, background=0.0).astype('float32')
        mask[ mask > 0] = 1
        nrrd.write(outdir+filename+'_mask.nrrd', mask, header=header)

        return mask


    def normalize_intensity(self, image):
        """Intensity normalization
        ---------------------------
        The intensity transformation maps integer values intensities in the interval [lmin, lmax] to integer values
        intensities in the interval [0,255]
        """
        lmin = float(image.min())
        lmax = float(image.max())
        return np.floor((image-lmin)/(lmax-lmin) * 255.)


    def hist_equalization(self, image):
        """Histogram equalization
        --------------------------
        Turn a non-uniform histogram of a low contrast image into a more uniform histogram
        Normalize cumulative histogram of the image
        1- first compute the intensity histogram
        2- second compute the corresponding cumuluative histogram H
        3- third compute the transformation to enhance the contrast
        """
        h = np.histogram(image.flatten(), bins=256)[0]
        H = np.cumsum(h) / float(np.sum(h))
        flat = image.flatten().astype('int')
        e = np.floor(H[flat] * 255.)
        return e.reshape(image.shape)




if __name__ == '__main__':
    path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
    os.makedirs(path + 'preprocessed_convexhull/', exist_ok=True)
    outdir = path+'preprocessed_convexhull/'
    #outdir_test = path+'preprocessed/test_masks/'
    pp = PreProcessing()
    pp.preprocess(path, outdir)
    #pp.create_mask(path + 'proc/20161102_32_C1_Scope_1_C1_down_result.nrrd', outdir=outdir_test)
    #preprocess.create_mask(path + 'proc/20161102_32_C1_Scope_1_C1_down_result.nrrd', outdir=outdir)
    #preprocess.create_mask(path + 'JRC2018_lo.nrrd')
    print('--- Done ---')
