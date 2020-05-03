###############################################################################
###############################################################################
###############    By: BRANDON CLINTON JONES
###############
###############     LAST UPDATED JAN 7 2020
###############
###############
############### SCRIPT TO CALL IMAGE AND MASK 3D GENERATOR AND LOOK AT OUTPUTS
###############
###############
###############################################################################
###############################################################################


pad_to_64 = True
zoomRange=(1,1) # around 1
rotationRange=0 # in degrees
horz_shift = 0 # % of the total number of pixels
vert_shift = 0
flipLR = False
flipUD = False
bool_shuffle = False

# pad_to_64 = True
# zoomRange=(0.9,1) # around 1
# rotationRange=10 # in degrees
# horz_shift = 0.1 # % of the total number of pixels
# vert_shift = 0.1
# flipLR = True
# flipUD = False
# bool_shuffle = False


import tensorflow
from custom_utils import csv_utils, old_custom_generators
from custom_utils.generator_viewer_3D import show_3d_images
from custom_utils.old_custom_generators import getFolderNamesFromDir, getFileNamesFromDir, getFileFrameNumber, getFileMappingForDir
from custom_utils.old_custom_generators import Generator_3D
from custom_utils.custom_generator_3D_PADTO64 import Generator_3D_PADTO64
import numpy as np
import time
from custom_utils.display import displayStack
import cv2


import os
import glob
from scipy.misc import imread

import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############            INPUTS        ###########################	
###############################################################################
###############################################################################


dirPath_train = "/d1/DeepLearning/3D_data/3D_data/train/"

dirPath_train_mask = (dirPath_train + "mask/")
dirPath_train_image = (dirPath_train + "image/")


dirPath_valid = "/d1/DeepLearning/3D_data/3D_data/valid/"

dirPath_valid_mask = (dirPath_train + "mask/")
dirPath_valid_image = (dirPath_train + "image/")


batch_size = 67
batch_size_valid = 95 - batch_size

seed = 1

numFramesPerStack = 60
total_train_images = 67
total_valid_images = 65 - total_train_images
# NUM FRAMES PER STACK IS THE TOTAL NUMBER OF SLICES 

img_size = 128

input_shape = ( 1, img_size , img_size , numFramesPerStack ) # (1, 200, 200, 352)
img_height = input_shape[ 1 ]
img_width = input_shape[ 2 ]
nChannels = input_shape[ 0 ]

num_classes = 1 # NOTE: FOR SEGMENTATION, SEEMS UNNECESSARY. USED ONLY FOR CLASSIFIER


###############################################################################
###############################################################################
###############            HELPER FUNCTIONS    ################################
###############################################################################
###############################################################################


## Function to combine mage and image generators into one for Keras training purposes

def combine_generator( gen, total_num_images, batch_size ):
    # Attempting to combine the generators instead of zipping them
    
    index=random.randint(0,total_num_images-batch_size)



    img , mask = gen.__getitem__(index)

    while True:
        yield ( img , mask )




###############################################################################
###############################################################################
###############     VALIDATION IMAGES    ######################################
###############################################################################
###############################################################################



fileIDList_valid_image, fileIDToPath_valid_image, fileIDToLabel_valid_image = getFileMappingForDir(dirPath_valid_image, numFramesPerStack)
fileIDList_length_valid_image = len(fileIDList_valid_image)





###############################################################################
###############################################################################
###############            LOADING FILES     ################################
###############################################################################
###############################################################################

fileIDList_train_image, fileIDToPath_train_image, fileIDToLabel_train_image = getFileMappingForDir(dirPath_train_image, numFramesPerStack)
fileIDList_length_train_image = len(fileIDList_train_image)



if pad_to_64:

    print('USING 64 PAD GENERATOR')
    print()
    print()

    train_image_generator = Generator_3D_PADTO64( fileIDList_train_image, fileIDToPath_train_image, 
        numFramesPerStack=numFramesPerStack, 
        batchSize = batch_size, 
        dim = ( img_height , img_width ) , nChannels = nChannels ,
        seed = seed , shuffle=bool_shuffle, sepToken="_", zoomRange=zoomRange, rotationRange=rotationRange, 
        widthShiftRange=vert_shift, heightShiftRange=horz_shift, 
        flipLR = flipLR, flipUD = flipUD )


    valid_image_generator = Generator_3D_PADTO64( fileIDList_valid_image , fileIDToPath_valid_image , 
        numFramesPerStack=numFramesPerStack, 
        batchSize = batch_size, 
        dim = ( img_height , img_width ) , nChannels = nChannels ,
        seed = seed , shuffle=bool_shuffle, sepToken="_", zoomRange=zoomRange, rotationRange=rotationRange, 
        widthShiftRange=vert_shift, heightShiftRange=horz_shift, 
        flipLR = flipLR, flipUD = flipUD )


else:

    print('USING 60 PAD GENERATOR')
    print()
    print()

    train_image_generator = Generator_3D( fileIDList_train_image, fileIDToPath_train_image, 
        numFramesPerStack=numFramesPerStack, 
        batchSize = batch_size, 
        dim = ( img_height , img_width ) , nChannels = nChannels ,
        seed = seed , shuffle=bool_shuffle, sepToken="_", zoomRange=zoomRange, rotationRange=rotationRange, 
        widthShiftRange=vert_shift, heightShiftRange=horz_shift, 
        flipLR = flipLR, flipUD = flipUD )









###############################################################################
###############################################################################
###############            VISUALIZATION CODE     #############################
###############################################################################
###############################################################################


train_generator = combine_generator( train_image_generator, total_train_images,
    batch_size )

valid_generator = combine_generator( valid_image_generator, total_valid_images,
    batch_size_valid )


index = 0


img_check, mask_check = valid_image_generator.__getitem__( index )


print(np.shape(img_check))
print(np.shape(mask_check))


# remove single dimension corresponding to channels
img_check = np.squeeze( img_check )
mask_check = np.squeeze( mask_check ) * 255



dummy_vars , img_h , img_w , num_slices = np.shape(img_check)

# Concatenate and store into one matrix for visualization:
display_img = np.zeros( ( batch_size , img_h , 2 * img_w , num_slices ) ) 



# Concatenate all into display_img
for cc in range(batch_size):
    print('Viewing image {} of 66'.format(cc))

    tmp_img = np.squeeze( img_check[ cc , : , : , : ] )
    tmp_mask = np.squeeze( mask_check[ cc , : , : , : ] )
    tmp = np.concatenate(( tmp_img, tmp_mask), axis=1)
    show_3d_images( tmp )



# Concatenate all into display_img
for cc in range(batch_size_valid):
	print('Viewing image {} of 28'.format(cc))

	tmp_img = np.squeeze( img_check[ cc , : , : , : ] )
	tmp_mask = np.squeeze( mask_check[ cc , : , : , : ] )
	tmp = np.concatenate(( tmp_img, tmp_mask), axis=1)
	show_3d_images( tmp )
