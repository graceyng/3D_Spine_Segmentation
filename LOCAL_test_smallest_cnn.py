# 3D Automated segmentation


import tensorflow
from custom_utils import custom_models
from custom_utils.custom_generators import Generator_3D
# import nibabel as nib
import string
import os
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint , TensorBoard
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
import csv
import json
import time
import random
import math
#import PIL
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import time


# Set CPU as available physical device
#my_devices = tensorflow.config.experimental.list_physical_devices(device_type='CPU')
#tensorflow.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

study_name = '3D_MRI_Spine_Segmentation'

#Parameters to select
metadataFile = '/Users/graceng/Documents/Med_School/Research/Radiology/043020_Metadata.json'
maskDir = 'collateVertMasks'
nChannels = 1
epochs = 1 #10
outputFile = 'trial1'

#Expected dictionary keys in metadata file
metadataVars = ['scanList_train', 'scanList_valid', 'dim', 'numFramesPerStack', 'dirPath', 'sliceFileExt', 'batch_size',
               'k_folds', 'k_idx', 'seed']
f = open(metadataFile, 'r')
metadata = json.load(f)
f.close()

for metadataVar in metadataVars:
    if metadataVar not in metadata:
        raise Exception('{} not in metadata file.'.format(metadataVar))
dim = tuple(metadata['dim'])

#TODO: switch back to full size
total_train_images = 10 #len(metadata['scanList_train'])
total_valid_images = len(metadata['scanList_valid'])

#Channels_first
#input_shape = (nChannels, dim[0], dim[1], metadata['numFramesPerStack'])

#Channels_last
input_shape = (dim[0], dim[1], metadata['numFramesPerStack'], nChannels)

# SO THE CHANNELS IS SET  TO ALWAYS FIRST OR ALWAYS LAST. IN THIS CASE, CHANNELS IS SET TO ALWAYS LAST
# SO THE TENSOR DIMENSIONS IN ORDER ARE IMG_HEIGHT , IMG_WIDTH , NUM_SLICES, NUM_CHANNELS

# NOTE: THE FIRST TWO DIMENSIONS OF INPUT SHAPE CAN BE CHANGED TO ARBITRARY NUMBERS.
#       THE CODE AUTOMATICALLY INTERPOLATES TO THAT NUMBER. 
#           HOWEVER, THE NUM_FRAMES_PER_STACK MUST NOT BE CHANGED


###############################################################################
###############################################################################
###############            TRAIN IMAGES    ####################################
###############################################################################
###############################################################################
#TODO: only using first 10 scans
train_generator = Generator_3D(metadata['scanList_train'][:10], metadata['dirPath'], maskDir,
                               numFramesPerStack=metadata['numFramesPerStack'], batchSize=metadata['batch_size'],
                               dim=dim, nChannels=nChannels, seed=metadata['seed'], shuffle=True,
                               sliceFileExt=metadata['sliceFileExt'], fitImgMethod="pad", zoomRange=(1,1),
                               rotationRange=0, widthShiftRange=0, heightShiftRange=0, flipLR=False, flipUD=False,
                               channel_order='last')

steps_per_epoch = train_generator.__len__()

###############################################################################
###############################################################################
###############     VALIDATION IMAGES    ######################################
###############################################################################
###############################################################################


valid_generator = Generator_3D(metadata['scanList_valid'], metadata['dirPath'], maskDir,
                               numFramesPerStack=metadata['numFramesPerStack'], batchSize=metadata['batch_size'],
                               dim=dim, nChannels=nChannels, seed=metadata['seed'], shuffle=True,
                               sliceFileExt=metadata['sliceFileExt'], fitImgMethod="pad", zoomRange=(1,1),
                               rotationRange=0, widthShiftRange=0, heightShiftRange=0, flipLR=False, flipUD=False,
                               channel_order='last')

validation_steps = valid_generator.__len__()

#TODO: see if this combine_generator function is necessary

def combine_generator( gen, total_num_images, batch_size ):

    counter = 0

    num_iterations = np.floor( total_num_images / batch_size ).astype( int )

    while True:
        for counter in range( num_iterations ):


            # index=random.randint(0,total_num_images-batch_size-1)
            index = counter * batch_size
            if index > total_num_images - batch_size - 1:
                index = total_num_images - batch_size - 1

            # print()  
            # print('Index number {}'.format(str(index)))
            
            img , mask = gen.__getitem__(index)
            yield ( img , mask )


train_generator = combine_generator( train_generator, total_train_images, metadata['batch_size'] )

valid_generator = combine_generator( valid_generator ,total_valid_images, metadata['batch_size'] )




###############################################################################
###############################################################################
###############     Helper Functions  ######################################
###############################################################################
###############################################################################

smooth = 1

def dice_coef(y_true, y_pred):
    #print(y_true.shape)
    #print(y_pred.shape)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    dice_coef_loss_out = 1 - dice_coef( y_true , y_pred )
    return dice_coef_loss_out 



###############################################################################
###############################################################################
###############     Model Section ######################################
###############################################################################
###############################################################################


# class WeightsRecorder(tensorflow.keras.callbacks.Callback):
class WeightsRecorder(tensorflow.keras.callbacks.Callback):
    def __init__(self, progressFilePath):
        super(WeightsRecorder, self).__init__()
        self.progressFilePath = progressFilePath
        self.lastTimePoint = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        training_loss = logs["loss"]
        validation_loss = logs["val_loss"]
        training_dice = logs["dice_coef"]
        validation_dice = logs["val_dice_coef"]
        ellapsed = "%0.1f" % (time.time() - self.lastTimePoint)
        self.lastTimePoint = time.time()

        with open(self.progressFilePath, "a") as outputFile: 
            writer = csv.DictWriter(outputFile, lineterminator='\n', fieldnames=["epoch","time(s)","training_loss","validation_loss", "training_dice","validation_dice"])
            writer.writerow({"epoch": epoch,"time(s)": ellapsed, "training_loss": training_loss,"validation_loss": validation_loss, "training_dice": training_dice, "validation_dice": validation_dice})


#Tensorflow 1.0
# config = tensorflow.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
config = tensorflow.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
tensorflow.keras.backend.set_session(tensorflow.Session(config=config))

"""
#Tensorflow 2.0
config = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
#tensorflow.config.threading.set_inter_op_parallelism_threads(0)
#tensorflow.config.threading.set_intra_op_parallelism_threads(0)
tensorflow.compat.v1.keras.backend.set_session(tensorflow.compat.v1.Session(config=config))
"""

#############################################
#############################################
#############################################
##### ///////////////////////////////////////
##### ///////////////////////////////////////
##### ///// 
##### ///// CALLBACKS
##### /////
##### ///////////////////////////////////////
##### ///////////////////////////////////////
#############################################
#############################################
#############################################


# File name must include batches, epochs, valid steps

# This section simply creates the output folder where the csv writer will save model info

outputDirPath = os.getcwd()+'/'+study_name+'/'+outputFile

if not os.path.exists(outputDirPath):
    os.makedirs(outputDirPath)
else:
    print('directory exists!')

tensorboardDirPath = os.getcwd()+'/'+study_name+'/TENSORBOARD/'+outputFile
print(tensorboardDirPath)

if not os.path.exists(tensorboardDirPath):
    os.makedirs(tensorboardDirPath)
else:
    print('directory exists!')


tensorboard_object = TensorBoard( log_dir=tensorboardDirPath ) 


# progressFilePath = os.path.join(outputDirPath, study_name+'_training_progress.csv')

# # If output csv file does not exist, it creates it
# if not os.path.isfile(progressFilePath):
#     with open(progressFilePath, "w") as outputFile: 
#         writer = csv.DictWriter(outputFile, lineterminator='\n', 
#             fieldnames=["epoch","time","training_loss","validation_loss", "training_dice","validation_dice"])
#         writer.writeheader()

# recorder = WeightsRecorder(progressFilePath)

weight_saver = ModelCheckpoint(os.path.join(outputDirPath, study_name+'.{epoch:02d}-{val_loss:.2f}-{val_dice_coef:.2f}.h5'),save_best_only=False, save_weights_only=False)

callbackList = [ weight_saver , tensorboard_object]

# callbackList = [recorder, weight_saver]
# # callbackList = [ recorder ]






### ***** THIS DEFINES THE CURRENT NEURAL NETWORK ARCHITECTURE ***** ###
# model = custom_models.get_simple_3d_unet(input_shape)
# model = custom_models.get_shallow_3d_unet( input_shape )
# model = custom_models.get_small_3d_unet( input_shape )
model = custom_models.get_test_3d_small_cnn( input_shape )
model.summary()

model.compile( optimizer=Adam(2e-5), loss=tensorflow.keras.losses.binary_crossentropy, 
    metrics=[dice_coef,dice_coef_loss] )

#  DEFINES WHICH METRICS WILL BE USED TO ANALYZE THE MODELS EFFICACY

#TODO: turned off multiprocessing
#TODO: test w/out validation
"""
hist=model.fit(train_generator, validation_data=valid_generator, validation_steps= validation_steps,
               steps_per_epoch= steps_per_epoch, epochs= epochs, verbose=1 ,  callbacks=callbackList,
               use_multiprocessing = 0, max_queue_size = 2)
"""
hist=model.fit(train_generator, steps_per_epoch= steps_per_epoch, epochs= epochs, verbose=1 ,  callbacks=callbackList,
               use_multiprocessing = 0, max_queue_size = 2)