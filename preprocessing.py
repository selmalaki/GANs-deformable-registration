
import scipy.ndimage.morphology as morph
import skimage.morphology as skimorph
from scipy.spatial import ConvexHull
from skimage import measure
from skimage import filters
from skimage import feature
from scipy import spatial
from scipy import ndimage
from scipy import misc
import nibabel as nib
import numpy as np
import ntpath
import nrrd
import os

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma


from joblib import Parallel, delayed
from multiprocessing import Pool

__author__ = 'elmalakis'

class PreProcessing():

    def __init__(self, path, outdir):
        self.path = path
        self.outdir = outdir

        self.filesdir = path + 'proc/'
        self.template = 'JRC2018_lo.nrrd'
        self.template_mask = None

        self.filelist = ['20161102_32_C1_Scope_1_C1_down_result.nrrd',
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

        self.allfiles = [self.filesdir+f for f in self.filelist]
        self.allfiles.append(path+self.template)


    def create_mask_train_examples(self):
        for f in self.filelist:
            self.create_mask(path=self.filesdir + f, outdir=self.outdir)
            print('--- Done ' + f + ' ---')

        self.create_template_mask()


    def create_template_mask(self):
        self.template_mask = self.create_mask(path=path+self.template,  outdir=self.outdir)
        print('--- Done template masking ---')
        return self.template_mask



    def create_sharpened_train_examples(self, cur_file):

        if self.template_mask is None:
            try:
                self.template_mask,_ = nrrd.read(self.path+self.template)
            except Exception as e:
                self.create_template_mask()

        cur_img, _ = nrrd.read(cur_file)
        den, _ = self.denoise_image(image=cur_img, mask=self.template_mask)
        sharp, diff = self.sharpening(image=den)

        nrrd.write(self.outdir + ntpath.basename(cur_file) + '_sharp.nrrd', sharp)
        nrrd.write(self.outdir + ntpath.basename(cur_file)  + '_sharp_diff.nrrd', sharp)
        print('finish sharpening: '+cur_file)


    def parallel_create_sharpened_train_examples(self):

        #print(self.allfiles)
        with Pool(16) as p:
            p.map(self.create_sharpened_train_examples, self.allfiles )



    def create_more_restricted_mask(self, path, outdir):
        """create a mask using otsu thresholding"""
        filename = os.path.basename(path)
        filename = os.path.splitext(filename)[0]

        image, header = nrrd.read(path)
        val = filters.threshold_otsu(image)
        blobs = image > val

        mask = morph.binary_dilation(blobs, iterations=10)
        mask = morph.binary_closing(mask, iterations=10)
        mask = morph.binary_fill_holes(mask)
        mask = measure.label(mask, background=0.0).astype('float32')
        mask[mask > 0] = 1
        nrrd.write(path + filename + '_mask_restricted.nrrd', mask, header=header)


    def create_mask(self, path, outdir):
        """create a mask using otsu thresholding"""
        filename = os.path.basename(path)
        filename = os.path.splitext(filename)[0]

        image, header = nrrd.read(path)
        image = self.normalize_intensity(image)
        image = self.hist_equalization(image)
        #nrrd.write(outdir+filename+ '_histogram_normalized.nrrd', image, header=header)

        #normalized_image = (image - np.mean(image)) / np.std(image)
        #nrrd.write(outdir + filename + '_normalized.nrrd', normalized_image, header=header)

        val = filters.threshold_otsu(image)
        # convert to binary to speed up next processing
        blobs = image > val
        # Dilation enlarges bright regions and shrinks dark regions
        mask = morph.binary_dilation(blobs, iterations=20)
        # Closing remove pepper spots and connects small bright cracks. Close up dark gaps between bright features
        mask = morph.binary_closing(mask, iterations=10)

        # Create a convex hull - Not the fastest way
        #for z in range(0, mask.shape[2]):
        #    mask[:,:,z] = skimorph.convex_hull_image(mask[:,:,z])

        #mask = spatial.ConvexHull(mask, incremental = True)
        # convert to 0 and 1 float
        mask = measure.label(mask, background=0.0).astype('float32')
        mask[ mask > 0] = 1
        nrrd.write(outdir+filename+'_dilated_mask.nrrd', mask, header=header)

        #print('--- Done ' + filename + ' ---')

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

    def denoise_image(self, image, mask):
        sigma = estimate_sigma(image, N=4)
        den = nlmeans(image, sigma=sigma, mask=mask, patch_radius=1, block_radius=1, rician=True)
        diff = np.abs(den.astype('f8') - image.astype('f8'))
        return den, diff

    def sharpening(self, image):
        blurred_f = ndimage.gaussian_filter(image, 3)
        filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        alpha = 30
        sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
        diff = np.abs(sharpened.astype('f8') - image.astype('f8'))
        return sharpened, diff


if __name__ == '__main__':
    path = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
    os.makedirs(path + 'preprocessed_convexhull/', exist_ok=True)
    outdir = path+'preprocessed_convexhull/'
    #outdir_test = path+'preprocessed/test_masks/'
    pp = PreProcessing(path, outdir)
    pp.parallel_create_sharpened_train_examples()
    #pp.create_mask(path + 'proc/20161102_32_C1_Scope_1_C1_down_result.nrrd', outdir=outdir_test)
    #preprocess.create_mask(path + 'proc/20161102_32_C1_Scope_1_C1_down_result.nrrd', outdir=outdir)
    #preprocess.create_mask(path + 'JRC2018_lo.nrrd')
    print('--- Done ---')
