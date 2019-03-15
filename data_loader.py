import scipy
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

class DataLoader():


    def __init__(self, batch_sz = 16):

        self.batch_sz = batch_sz
        self.n_gpus = 1
        self.crop_sz = (40, 40, 40)  # the image shape is 1227,1996,40, and sometimes 1170, 1996,43
        self.mask_sz = (40, 40, 40)

        self.imgs = []
        self.masks = []
        self.img_template = None
        self.mask_template = None

        self.imgs_test = []
        self.masks_test = []

        filepath = '/groups/scicompsoft/home/elmalakis/Work/Janelia/ImageRegistration/data/for_salma/preprocess_to_4/'
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
            # images
            curr_img, meta_dict = self._read_nifti(img_pp[i])
            curr_img = np.float32(curr_img)
            curr_img = (curr_img - np.mean(curr_img))/ np.std(curr_img)
            self.imgs.append(curr_img)
            # masks
            curr_mask, meta_dict = self._read_nifti(mask_pp[i])
            curr_mask = np.float32(curr_mask)
            self. masks.append(curr_mask)

        # template is subject 4
        self.img_template = self.imgs.pop(3)
        self.mask_template = self.masks.pop(3)

        # test is subject 16, 17, 18 after popping subject 4 the indices are 14, 15, 16 (Indices start at 0)
        self.imgs_test.append(self.imgs.pop(16))
        self.imgs_test.append(self.imgs.pop(15))
        self.imgs_test.append(self.imgs.pop(14))

        self.masks_test.append(self.masks.pop(16))
        self.masks_test.append(self.masks.pop(15))
        self.masks_test.append(self.masks.pop(14))


    def get_template(self):
        return self.img_template

    def load_data(self, batch_size=1, is_testing=False):
        idx = 0
        if is_testing:
            #ue batch_size=1 for now
            idx, batch_images = np.random.choice(list(enumerate(self.imgs_test)))
        else:
            batch_images = np.random.choice(self.imgs, size=batch_size)

        return idx, batch_images


    def load_batch(self):
        while True:
            print('----- loading a batch -----')
            batch_img = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1), dtype='float32')
            batch_mask = np.zeros((self.batch_sz, self.mask_sz[0], self.mask_sz[1], self.mask_sz[2], 1), dtype='float32')

            batch_img_template = np.zeros((self.batch_sz, self.crop_sz[0], self.crop_sz[1], self.crop_sz[2], 1), dtype='float32')
            batch_mask_template = np.zeros((self.batch_sz, self.mask_sz[0], self.mask_sz[1], self.mask_sz[2], 1), dtype='float32')

            # randomly crop an image from imgs list
            idx = np.random.randint(0, len(self.imgs))
            img_for_crop = self.imgs[idx]
            mask_for_crop = self.masks[idx]

            num_crop = 0
            while num_crop < self.batch_sz:
                x = np.random.randint(0, img_for_crop.shape[0] - self.crop_sz[0])
                y = np.random.randint(0, img_for_crop.shape[1] - self.crop_sz[1])
                z = 0 # take the whole dimension of Z
                # crop in the x-y dimension only and use the all the slices
                cropped_img = img_for_crop[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                cropped_img_template = self.img_template[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                cropped_mask = mask_for_crop[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                cropped_mask_template = self.mask_template[x:x + self.crop_sz[0], y:y + self.crop_sz[1], z:z+self.crop_sz[2]]
                # if include the random crop in training
                is_include = False
                num_vox = len(cropped_mask[cropped_mask == 255])

                accept_prob = np.random.random()
                if num_vox > 500 and accept_prob > 0.98:
                    is_include = True

                if is_include:
                    batch_img[num_crop,:,:,:,0] = cropped_img
                    batch_mask[num_crop,:,:,:,0] = cropped_mask

                    batch_img_template[num_crop,:,:,:,0] = cropped_img_template
                    batch_mask_template[num_crop,:,:,:,0] = cropped_mask_template

                    num_crop += 1

            # TODO: data augmentation if needed

            yield batch_img, batch_img_template


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


