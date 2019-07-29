import nrrd
import scipy
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.transform import resize
#locally
#import ImageRegistrationGANs.preprocessing as pp

# on cluster
import preprocessing as pp

__author__ = 'elmalakis'


DEBUG = 1

class DataLoader():

    def __init__(self,
                 batch_sz = 16,
                 dataset_name ='fly',
                 crop_size= (64, 64, 64),
                 use_hist_equilized_data = False,
                 min_max = False,
                 restricted_mask=False,
                 use_golden=False,
                 use_sharpen=False):
        """
        :param batch_sz: int - size of the batch
        :param sampletype: string - 'fly' or 'fish'
        """
        self.batch_sz = batch_sz
        self.crop_sz = crop_size

        self.imgs = []
        self.masks = []
        self.img_template = None
        self.mask_template = None

        self.imgs_test = []
        self.masks_test = []

        self.use_golden = use_golden
        self.use_sharpen = use_sharpen

        if dataset_name is 'fly':
            self.imgs, self.masks, self.img_template, self.mask_template, self.imgs_test, self.masks_test, self.golden_imgs, self.n_batches = self.prepare_fly_data(batch_sz, use_hist_equilized_data, min_max, restricted_mask, use_golden, use_sharpen)
        elif dataset_name is 'fish':
            self.imgs, self.masks, self.img_template, self.mask_template, self.imgs_test, self.masks_test, self.n_batches = self.prepare_fish_data(batch_sz)
        elif dataset_name is 'toy':
            self.imgs, self.img_template, self.n_batches = self.prepare_toy_data(batch_sz)
        else:
            raise ValueError('Data of type %s is not available' % (dataset_name))



    def prepare_fish_data(self, batch_sz):

        self.batch_sz = batch_sz
        self.n_gpus = 1
        #self.crop_sz = (self.crop_sz, self.crop_sz, self.crop_sz)  # the image shape is 1227,1996,40, and sometimes 1170, 1996,43
        self.mask_sz = self.crop_sz #(self.crop_sz, self.crop_sz, self.crop_sz)

        imgs = []
        masks = []
        img_template = None
        mask_template = None

        imgs_test = []
        masks_test = []

        template_shape = (1166, 1996, 40) # most of the images have this size

        filepath = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/fishdata/data/for_salma/preprocess_to_4/'
        img_pp = [filepath +'subject_1_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_2_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_3_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_4_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_5_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_6_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_7_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_8_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_9_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_10_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_11_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_12_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_13_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_14_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_15_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_16_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_17_anat_stack_regiprep_pp.nii.gz',
                  filepath + 'subject_18_anat_stack_regiprep_pp.nii.gz'
                  ]

        mask_pp = [filepath + 'subject_1_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_2_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_3_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_4_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_5_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_6_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_7_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_8_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_9_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_10_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_11_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_12_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_13_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_14_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_15_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_16_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_17_anat_stack_regiprep_mask.nii.gz',
                   filepath + 'subject_18_anat_stack_regiprep_mask.nii.gz'
                  ]

        print('----- loading data file -----')
        for i in range(len(img_pp)):
            # images normalize
            curr_img, meta_dict = self._read_nifti(img_pp[i])
            curr_img = np.float32(curr_img)
            curr_img = (curr_img - np.mean(curr_img))/ np.std(curr_img)
            # masks 1: interesting value, 0: not interesting value
            curr_mask, meta_dict = self._read_nifti(mask_pp[i])
            curr_mask = np.float32(curr_mask)

            # resize
            if curr_img.shape != template_shape:
                curr_img = resize(curr_img, template_shape, anti_aliasing=True)
                curr_mask = resize(curr_mask, template_shape, anti_aliasing=True)

            masks.append(curr_mask)
            imgs.append(curr_img)

        # template is subject 4
        img_template = self.imgs.pop(3)
        mask_template = self.masks.pop(3)

        # test is subject 16, 17, 18 after popping subject 4 the indices are 14, 15, 16 (Indices start at 0)
        imgs_test.append(self.imgs.pop(16))
        imgs_test.append(self.imgs.pop(15))
        imgs_test.append(self.imgs.pop(14))

        masks_test.append(self.masks.pop(16))
        masks_test.append(self.masks.pop(15))
        masks_test.append(self.masks.pop(14))

        n_batches = int( (len(self.imgs) * template_shape[0] / self.crop_sz[0] )  / self.batch_sz)

        return imgs, masks, img_template, mask_template, imgs_test, masks_test, n_batches


    def prepare_fly_data(self, batch_sz, use_hist_equilized_data=False, min_max=False, restricted_mask=False, use_golden=False, use_sharpen=False):
        self.batch_sz = batch_sz
        self.n_gpus = 1
        self.mask_sz = self.crop_sz

        imgs = []
        masks = []
        golden_imgs = []
        #preprocess = pp.PreProcessing()

        img_template = None
        mask_template = None

        imgs_test = []
        masks_test = []

        filepath = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/flydata/forSalma/lo_res/'
        goldenfilepath = '/nrs/saalfeld/john/public/forSalma/lo_res/proc/'
        if use_hist_equilized_data:
            img_pp_normalized = [filepath + 'preprocessed_convexhull/' + '20161102_32_C1_Scope_1_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161102_32_C3_Scope_4_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161102_32_D1_Scope_1_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161102_32_D2_Scope_1_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161102_32_E1_Scope_1_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161102_32_E3_Scope_4_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161220_31_I1_Scope_2_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161220_31_I2_Scope_6_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161220_31_I3_Scope_6_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161220_32_C1_Scope_3_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20161220_32_C3_Scope_3_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20170223_32_A2_Scope_3_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20170223_32_A3_Scope_3_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20170223_32_A6_Scope_2_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20170223_32_E1_Scope_3_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20170223_32_E2_Scope_3_C1_down_result_histogram_normalized.nrrd',
                                 filepath + 'preprocessed_convexhull/' + '20170223_32_E3_Scope_3_C1_down_result_histogram_normalized.nrrd'
                                #filepath + 'preprocessed_convexhull/' + '20170301_31_B1_Scope_1_C1_down_result_histogram_normalized.nrrd', # remove the last 3 for testing
                                #filepath + 'preprocessed_convexhull/' + '20170301_31_B3_Scope_1_C1_down_result_histogram_normalized.nrrd',
                                #filepath + 'preprocessed_convexhull/' + '20170301_31_B5_Scope_1_C1_down_result_histogram_normalized.nrrd'
                      ]

        else:
            img_pp = [filepath + 'proc/' + '20161102_32_C1_Scope_1_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161102_32_C3_Scope_4_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161102_32_D1_Scope_1_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161102_32_D2_Scope_1_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161102_32_E1_Scope_1_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161102_32_E3_Scope_4_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161220_31_I1_Scope_2_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161220_31_I2_Scope_6_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161220_31_I3_Scope_6_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161220_32_C1_Scope_3_C1_down_result.nrrd',
                      filepath + 'proc/' + '20161220_32_C3_Scope_3_C1_down_result.nrrd',
                      filepath + 'proc/' + '20170223_32_A2_Scope_3_C1_down_result.nrrd',
                      filepath + 'proc/' + '20170223_32_A3_Scope_3_C1_down_result.nrrd',
                      filepath + 'proc/' + '20170223_32_A6_Scope_2_C1_down_result.nrrd',
                      filepath + 'proc/' + '20170223_32_E1_Scope_3_C1_down_result.nrrd',
                      filepath + 'proc/' + '20170223_32_E2_Scope_3_C1_down_result.nrrd',
                      filepath + 'proc/' + '20170223_32_E3_Scope_3_C1_down_result.nrrd'
                     #filepath + 'proc/' + '20170301_31_B1_Scope_1_C1_down_result.nrrd', # remove the last 3 for testing
                     #filepath + 'proc/' + '20170301_31_B3_Scope_1_C1_down_result.nrrd',
                     #filepath + 'proc/' + '20170301_31_B5_Scope_1_C1_down_result.nrrd'
                      ]


        mask_pp = [filepath + 'preprocessed_convexhull/' +'20161102_32_C1_Scope_1_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_C3_Scope_4_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_D1_Scope_1_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_D2_Scope_1_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_E1_Scope_1_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_E3_Scope_4_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_31_I1_Scope_2_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_31_I2_Scope_6_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_31_I3_Scope_6_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_32_C1_Scope_3_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_32_C3_Scope_3_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_A2_Scope_3_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_A3_Scope_3_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_A6_Scope_2_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_E1_Scope_3_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_E2_Scope_3_C1_down_result_mask.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_E3_Scope_3_C1_down_result_mask.nrrd'
                  #filepath + 'preprocessed_convexhull/' +'20170301_31_B1_Scope_1_C1_down_result_mask.nrrd', # remove the last 3 for testing
                  #filepath + 'preprocessed_convexhull/' +'20170301_31_B3_Scope_1_C1_down_result_mask.nrrd',
                  #filepath + 'preprocessed_convexhull/' +'20170301_31_B5_Scope_1_C1_down_result_mask.nrrd'
                   ]

        golden = [goldenfilepath + '20161102_32_C1_Scope_1_C1_down/result.0.nrrd',
                  goldenfilepath + '20161102_32_C3_Scope_4_C1_down/result.0.nrrd',
                  goldenfilepath + '20161102_32_D1_Scope_1_C1_down/result.0.nrrd',
                  goldenfilepath + '20161102_32_D2_Scope_1_C1_down/result.0.nrrd',
                  goldenfilepath + '20161102_32_E1_Scope_1_C1_down/result.0.nrrd',
                  goldenfilepath + '20161102_32_E3_Scope_4_C1_down/result.0.nrrd',
                  goldenfilepath + '20161220_31_I1_Scope_2_C1_down/result.0.nrrd',
                  goldenfilepath + '20161220_31_I2_Scope_6_C1_down/result.0.nrrd',
                  goldenfilepath + '20161220_31_I3_Scope_6_C1_down/result.0.nrrd',
                  goldenfilepath + '20161220_32_C1_Scope_3_C1_down/result.0.nrrd',
                  goldenfilepath + '20161220_32_C3_Scope_3_C1_down/result.0.nrrd',
                  goldenfilepath + '20170223_32_A2_Scope_3_C1_down/result.0.nrrd',
                  goldenfilepath + '20170223_32_A3_Scope_3_C1_down/result.0.nrrd',
                  goldenfilepath + '20170223_32_A6_Scope_2_C1_down/result.0.nrrd',
                  goldenfilepath + '20170223_32_E1_Scope_3_C1_down/result.0.nrrd',
                  goldenfilepath + '20170223_32_E2_Scope_3_C1_down/result.0.nrrd',
                  goldenfilepath + '20170223_32_E3_Scope_3_C1_down/result.0.nrrd'
                 # goldenfilepath + '20170301_31_B1_Scope_1_C1_down/result.0.nrrd',
                 # goldenfilepath + '20170301_31_B3_Scope_1_C1_down/result.0.nrrd',
                 # goldenfilepath +  '20170301_31_B5_Scope_1_C1_down/result.0.nrrd'
                ]


        sharpen = [filepath + 'preprocessed_convexhull/' +'20161102_32_C1_Scope_1_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_C3_Scope_4_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_D1_Scope_1_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_D2_Scope_1_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_E1_Scope_1_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161102_32_E3_Scope_4_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_31_I1_Scope_2_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_31_I2_Scope_6_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_31_I3_Scope_6_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_32_C1_Scope_3_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20161220_32_C3_Scope_3_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_A2_Scope_3_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_A3_Scope_3_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_A6_Scope_2_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_E1_Scope_3_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_E2_Scope_3_C1_down_result_sharp.nrrd',
                   filepath + 'preprocessed_convexhull/' +'20170223_32_E3_Scope_3_C1_down_result_sharp.nrrd'
                  #filepath + 'preprocessed_convexhull/' +'20170301_31_B1_Scope_1_C1_down_result_sharp.nrrd', # remove the last 3 for testing
                  #filepath + 'preprocessed_convexhull/' +'20170301_31_B3_Scope_1_C1_down_result_sharp.nrrd',
                  #filepath + 'preprocessed_convexhull/' +'20170301_31_B5_Scope_1_C1_down_result_sharp.nrrd'
                   ]
        sharpen_diff = [filepath + 'preprocessed_convexhull/' +'20161102_32_C1_Scope_1_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161102_32_C3_Scope_4_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161102_32_D1_Scope_1_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161102_32_D2_Scope_1_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161102_32_E1_Scope_1_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161102_32_E3_Scope_4_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161220_31_I1_Scope_2_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161220_31_I2_Scope_6_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161220_31_I3_Scope_6_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161220_32_C1_Scope_3_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20161220_32_C3_Scope_3_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20170223_32_A2_Scope_3_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20170223_32_A3_Scope_3_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20170223_32_A6_Scope_2_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20170223_32_E1_Scope_3_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20170223_32_E2_Scope_3_C1_down_result_sharp_diff.nrrd',
                        filepath + 'preprocessed_convexhull/' +'20170223_32_E3_Scope_3_C1_down_result_sharp_diff.nrrd'
                      # filepath + 'preprocessed_convexhull/' +'20170301_31_B1_Scope_1_C1_down_result_sharp_diff.nrrd', # remove the last 3 for testing
                      # filepath + 'preprocessed_convexhull/' +'20170301_31_B3_Scope_1_C1_down_result_sharp_diff.nrrd',
                      # filepath + 'preprocessed_convexhull/' +'20170301_31_B5_Scope_1_C1_down_result_sharp_diff.nrrd'
                   ]
        # Template Preparation

        # The original template
        img_template, templ_header = nrrd.read(filepath + 'JRC2018_lo.nrrd')
        mask_template, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/JRC2018_lo_dilated_mask.nrrd')

        # Use one of the training data as a template
        #img_template, templ_header = nrrd.read(filepath + 'proc/' + '20170223_32_A3_Scope_3_C1_down_result.nrrd')
        #mask_template, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/20170223_32_A3_Scope_3_C1_down_result_dilated_mask.nrrd')

        # Use the sharpened version as a template
        if sharpen:
            img_template, templ_header =  nrrd.read(filepath + 'preprocessed_convexhull/JRC2018_lo_sharp.nrrd')
            img_template_diff, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/JRC2018_lo_sharp_diff.nrrd')

        img_template = np.float32(img_template)
        mask_template = np.float32(mask_template)

        # Apply the template mask before the standardization
        mask = 1 - mask_template
        mask = np.uint8(mask)

        img_template_masked = np.ma.array(img_template, mask=mask)
        img_template = (img_template - np.mean(img_template_masked)) / np.std(img_template_masked)
        if min_max:
            img_template = (2 * (img_template - np.min(img_template_masked)) / (
                    np.max(img_template_masked) - np.min(img_template_masked))) - 1

        # nrrd.write(filepath + '20170223_32_A3_Scope_3_C1_down_result_undermask.nrrd', img_template,
        #            header=templ_header)


        if use_golden:

            for g in golden:
                g_img, g_header = nrrd.read(g)
                g_img = np.float32(g_img)

                # image standardization
                img_masked = np.ma.array(g_img, mask=mask)
                g_img = (g_img - img_masked.mean())/img_masked.std()

                # image normalization
                if min_max:
                    g_img = (2 * (g_img - np.min(img_masked)) / (np.max(img_masked) - np.min(img_masked))) - 1

                golden_imgs.append(g_img)


        elif use_hist_equilized_data:
            # histogram equalized template
            img_template, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/' +'JRC2018_lo_histogram_normalized.nrrd')
            #mask_template, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/' + 'JRC2018_lo_dilated_mask.nrrd')
            #if restricted_mask:
            #    mask_template, templ_header = nrrd.read(filepath + 'JRC2018_lo_mask_restricted.nrrd')

            # For testing use template from the actual dataset (20170223_32_A3_Scope_3_C1_down_result)
            #img_template, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/' +'20170223_32_A3_Scope_3_C1_down_result_histogram_normalized.nrrd')
            #mask_template, templ_header = nrrd.read(filepath + 'preprocessed_convexhull/' +'20170223_32_A3_Scope_3_C1_down_result_dilated_mask.nrrd')

            img_template = np.float32(img_template)
            #mask_template = np.float32(mask_template)

            # Apply the template mask before the normalization
            img_template_masked = np.ma.array(img_template, mask=mask)
            img_template = (img_template - np.mean(img_template_masked)) / np.std(img_template_masked)
            if min_max:
                img_template = (2 * (img_template - np.min(img_template_masked)) / (np.max(img_template_masked) - np.min(img_template_masked))) - 1


            print('----- loading histogram equalized data files -----')
            for inorm in img_pp_normalized:
                # images normalize
                curr_img, img_header = nrrd.read(inorm)
                curr_img = np.float32(curr_img)

                # Apply the template mask before standardization
                mask = 1-mask_template
                mask = np.uint8(mask)
                img_masked = np.ma.array(curr_img, mask=mask)
                curr_img = (curr_img - img_masked.mean())/img_masked.std()

                # image normalization
                if min_max:
                    curr_img = (2 * (curr_img - np.min(img_masked)) / (np.max(img_masked) - np.min(img_masked))) - 1

                # Just use the template mask
                # masks 1: interesting value, 0: not interesting value
                #curr_mask, mask_header = nrrd.read(mask_pp[i])
                #curr_mask = np.float32(curr_mask)
                #masks.append(curr_mask)
                imgs.append(curr_img)


        elif use_sharpen:
            print('----- loading sharpened files -----')
            for ish in sharpen:
                curr_img, img_header = nrrd.read(ish)
                curr_img = np.float32(curr_img)

                mask = 1-mask_template
                mask = np.uint8(mask)
                img_masked = np.ma.array(curr_img, mask=mask)

                # image standardization
                curr_img = (curr_img - img_masked.mean()) / img_masked.std()

                # image normalization
                if min_max:
                    curr_img = (2 * (curr_img - np.min(img_masked)) / (np.max(img_masked) - np.min(img_masked))) - 1

                imgs.append(curr_img)
        else:
            print('----- loading normal data files -----')
            for ipp in img_pp:
                curr_img, img_header = nrrd.read(ipp)
                curr_img = np.float32(curr_img)
                # denoise
                #den, _ = preprocess.denoise_image(image=curr_img, mask=mask_template)
                #sharp, _ = preprocess.sharpening(image=den)
                # Apply the template mask before the standardization
                mask = 1 - mask_template
                mask = np.uint8(mask)
                img_masked = np.ma.array(curr_img, mask=mask)
                curr_img = (curr_img - np.mean(img_masked)) / np.std(img_masked)
                # image normalization
                if min_max:
                    curr_img = (2* (curr_img - np.min(img_masked))/(np.max(img_masked)-np.min(img_masked))) -1
                # masks 1: interesting value, 0: not interesting value
                #curr_mask, mask_header = nrrd.read(mask_pp[i])
                #curr_mask = np.float32(curr_mask)
                #masks.append(curr_mask)
                imgs.append(curr_img)



        # TODO: save test images
        # test is subject 16, 17, 18 after popping subject 4 the indices are 14, 15, 16 (Indices start at 0)
        # imgs_test.append(self.imgs.pop(16))
        # imgs_test.append(self.imgs.pop(15))
        # imgs_test.append(self.imgs.pop(14))
        #
        # masks_test.append(self.masks.pop(16))
        # masks_test.append(self.masks.pop(15))
        # masks_test.append(self.masks.pop(14))

        n_batches = int((len(imgs) * img_template.shape[0] / self.crop_sz[0]) / self.batch_sz)

        return imgs, masks, img_template, mask_template, imgs_test, masks_test, golden_imgs, n_batches


    def prepare_toy_data(self, batch_sz):
        imgs = []
        filepath = '/nrs/scicompsoft/elmalakis/GAN_Registration_Data/ToyExample/WarpedSpheres/'
        for i in range(50):
            curr_img, _ =  nrrd.read(filepath + str(i) + '/warped.nrrd')
            curr_img = np.float32(curr_img)
            curr_img = (curr_img - np.mean(curr_img)) / np.std(curr_img)
            imgs.append(curr_img)
        template, _ = nrrd.read(filepath + 'sphere.nrrd')
        template = np.float32(template)
        template = (template - np.mean(template)) / np.std(template)
        n_batches = int(50*template.shape[0]/self.crop_sz[0]/self.batch_sz)
        return imgs, template, n_batches


    def get_template(self):
        return self.img_template

    def load_data(self, batch_size=1, is_testing=False, is_validation=False):
        #idxs = []
        #test_images = []
        #test_masks = []

        #for batch in range(batch_size):

        idx = None
        test_image = None
        test_mask = None
        if is_testing:
             #use batch_size=1 for now
            idx, test_image = random.choice(list(enumerate(self.imgs_test)))
            test_mask = self.masks_test[idx]
            #idxs.append(idx)
            #test_images.append(test_image)
            #test_masks.append(test_mask)

        elif is_validation:
            idx, test_image = random.choice(list(enumerate(self.imgs)))

        return idx, test_image#, test_mask


    def load_batch_toy(self):
        for i in range(self.n_batches - 1):
            batch_img = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1), dtype='float32')
            batch_img_template = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1),dtype='float32')

            num_imgs = 0
            id = []

            while num_imgs < self.batch_sz:
                idx = np.random.randint(0, len(self.imgs))
                id.append(idx)
                chosen_img = self.imgs[idx]
                batch_img[num_imgs,:,:,:,0] = chosen_img
                batch_img_template[num_imgs,:,:,:,0] = self.img_template

                num_imgs += 1

            x_flip = np.random.randint(2, size=self.batch_sz)
            z_flip = np.random.randint(2, size=self.batch_sz)
            rot_angle = np.random.randint(4, size=self.batch_sz)
            for j in range(self.batch_sz):
                if x_flip[j]:
                    batch_img[j, :, :, :, 0] = np.flip(batch_img[j, :, :, :, 0], axis=0)
                    batch_img_template[j, :, :, :, 0] = np.flip(batch_img_template[j, :, :, :, 0], axis=0)
                if z_flip[j]:
                    batch_img[j, :, :, :, 0] = np.flip(batch_img[j, :, :, :, 0], axis=2)
                    batch_img_template[j, :, :, :, 0] = np.flip(batch_img_template[j, :, :, :, 0], axis=2)
                if rot_angle[j]:
                    batch_img[j, :, :, :, 0] = np.rot90(batch_img[j, :, :, :, 0], rot_angle[j], axes=(0, 1))
                    batch_img_template[j, :, :, :, 0] = np.rot90(batch_img_template[j, :, :, :, 0], rot_angle[j], axes=(0, 1))
            yield batch_img, batch_img_template, id



    def load_batch(self, dataset_name ='fly'):

        for i in range(self.n_batches - 1):
            #print('----- loading a batch -----')
            batch_img = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1), dtype='float32')
            #batch_mask = np.zeros((self.batch_sz, self.mask_sz[0], self.mask_sz[1], self.mask_sz[2], 1), dtype='float32')

            batch_img_template = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1), dtype='float32')
            #batch_mask_template = np.zeros((self.batch_sz, self.mask_sz[0], self.mask_sz[1], self.mask_sz[2], 1), dtype='float32')

            batch_img_golden = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1), dtype='float32')

            # randomly crop an image from imgs list
            idx = np.random.randint(0, len(self.imgs))
            img_for_crop = self.imgs[idx]
            if self.use_golden: golden_for_crop = self.golden_imgs[idx]
            #mask_for_crop = self.masks[idx]

            num_crop = 0
            while num_crop < self.batch_sz:
                x = np.random.randint(0, img_for_crop.shape[0] - self.crop_sz[0])
                y = np.random.randint(0, img_for_crop.shape[1] - self.crop_sz[1])
                if dataset_name is 'fish': z = 0 # take the whole dimension of Z
                else: z = np.random.randint(0, img_for_crop.shape[2] - self.crop_sz[2])
                # crop in the x-y dimension only and use the all the slices for fish
                cropped_img = img_for_crop[x:x+self.crop_sz[0], y:y+self.crop_sz[1], z:z+self.crop_sz[2]]
                cropped_img_template = self.img_template[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                if self.use_golden: cropped_img_golden = golden_for_crop[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                #cropped_mask = mask_for_crop[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                cropped_mask_template = self.mask_template[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                # if include the random crop in training
                is_include = False
                #num_vox = len(cropped_mask[cropped_mask == 1])
                num_vox = len(cropped_mask_template[cropped_mask_template == 1]) # Use the template mask instead for all the files

                if (num_vox/len(cropped_mask_template)) > 0.90: # 75% of the crop is under the mask
                    is_include = True

                #accept_prob = np.random.random()
                # if num_vox > 500: #and accept_prob > 0.98:
                #     is_include = True

                if is_include:
                    #if DEBUG: print('include this batch %d, %d, %d' %(x, y, z))
                    batch_img[num_crop,:,:,:,0] = cropped_img
                    #batch_mask[num_crop,:,:,:,0] = cropped_mask

                    # filter the image with the mask
                    # batch_img = batch_img * batch_mask

                    batch_img_template[num_crop,:,:,:,0] = cropped_img_template
                    #batch_mask_template[num_crop,:,:,:,0] = cropped_mask_template

                    # filter the template with the mask
                    #batch_img_template = batch_img_template * batch_mask_template

                    if self.use_golden: batch_img_golden[num_crop,:,:,:,0] = cropped_img_golden

                    num_crop += 1

            # data augmentation
            x_flip = np.random.randint(2, size=self.batch_sz)
            z_flip = np.random.randint(2, size=self.batch_sz)
            rot_angle = np.random.randint(4, size=self.batch_sz)
            for j in range(self.batch_sz):
                if x_flip[j]:
                    batch_img[j, :, :, :, 0] = np.flip(batch_img[j, :, :, :, 0], axis=0)
                    #batch_mask[j, :, :, :, 0] = np.flip(batch_mask[j, :, :, :, 0], axis=0)
                    batch_img_template[j, :, :, :, 0] = np.flip(batch_img_template[j, :, :, :, 0], axis=0)
                    #batch_mask_template[j, :, :, :, 0] = np.flip(batch_mask_template[j, :, :, :, 0], axis=0)
                    if self.use_golden: batch_img_golden[j, :, :, :, 0] = np.flip(batch_img_golden[j, :, :, :, 0], axis=0)
                if z_flip[j]:
                    batch_img[j, :, :, :, 0] = np.flip(batch_img[j, :, :, :, 0], axis=2)
                    #batch_mask[j, :, :, :, 0] = np.flip(batch_mask[j, :, :, :, 0], axis=2)
                    batch_img_template[j, :, :, :, 0] = np.flip(batch_img_template[j, :, :, :, 0], axis=2)
                    #batch_mask_template[j, :, :, :, 0] = np.flip(batch_mask_template[j, :, :, :, 0], axis=2)
                    if self.use_golden: batch_img_golden[j, :, :, :, 0] = np.flip(batch_img_golden[j, :, :, :, 0], axis=2)
                if rot_angle[j]:
                    batch_img[j, :, :, :, 0] = np.rot90(batch_img[j, :, :, :, 0], rot_angle[j], axes=(0, 1))
                    #batch_mask[j, :, :, :, 0] = np.rot90(batch_mask[j, :, :, :, 0], rot_angle[j], axes=(0, 1))
                    batch_img_template[j, :, :, :, 0] = np.rot90(batch_img_template[j, :, :, :, 0], rot_angle[j], axes=(0, 1))
                    #batch_mask_template[j, :, :, :, 0] = np.rot90(batch_mask_template[j, :, :, :, 0], rot_angle[j], axes=(0, 1))
                    if self.use_golden: batch_img_golden[j, :, :, :, 0] = np.rot90(batch_img_golden[j, :, :, :, 0], rot_angle[j], axes=(0, 1))

            yield batch_img, batch_img_template, batch_img_golden


    def _read_nifti(self,path, meta_dict={}):
        import nibabel as nib
        image = nib.load(path)
        image_data = image.get_data().squeeze()
        new_meta_dict = dict(image.header)
        meta_dict = {**new_meta_dict, **meta_dict}
        return image_data, meta_dict

    def _write_nifti(self,path, image_data, meta_dict={}):
        #    image_data = _standardize_axis_order(image_data, 'nii') # possibly a bad idea
        import nibabel as nib
        image = nib.Nifti1Image(image_data, None)
        for key in meta_dict.keys():
            if key in image.header.keys():
                image.header[key] = meta_dict[key]
        nib.save(image, path)


if __name__ == '__main__':
    dataloader = DataLoader()
    dataloader.load_batch()