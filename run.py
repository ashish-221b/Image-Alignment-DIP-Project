import numpy as np
import sample
#def get_transform( u, v, n, t, alp) :

im_1 = 'TestImages/test_mri.jpg'
# im_2 = 'TestImages/test_mri_translate.jpg'
# im_2 = 'TestImages/test_mri_rot2.jpg'
im_2 = 'TestImages/test_mri_scale.jpg'
# im_2 = 'TestImages/test_mri_shear.jpg'
print(sample.get_transform( im_1, im_2, 1000, 50, 0.0002))
