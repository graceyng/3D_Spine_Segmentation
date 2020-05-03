# 3D Automated segmentation for SPGR proximal femur images


import tensorflow
from custom_utils import csv_utils, old_custom_generators
from custom_utils import custom_models
from custom_utils.old_custom_generators import getFolderNamesFromDir, getFileNamesFromDir, getFileFrameNumber, getFileMappingForDir
# from custom_utils.custom_generators import Generator3DClassifier
from custom_utils.old_custom_generators import Generator3D
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
import time
import random
import math
import PIL
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import time


# NAME = 'FIRST_TENSORBOARD_ATTEMPT-{}'.format( int( time.time() ) )

study_name = '3D_SPGR_Segmentation'


dirPath_train = "/d1/DeepLearning/3D_data/3D_data/train/"

dirPath_valid = "/d1/DeepLearning/3D_data/3D_data/valid/"

dirPath_train_mask = (dirPath_train + "mask/")
dirPath_train_image = (dirPath_train + "image/")
dirPath_valid_mask = (dirPath_valid + "mask/")
dirPath_valid_image = (dirPath_valid + "image/")



batch_size = 2 # 32
epochs = 20 #10 

total_images = 95
total_train_images = 67
total_valid_images = total_images - total_train_images

steps_per_epoch = np.ceil( total_train_images / batch_size )
validation_steps = np.ceil( total_valid_images / batch_size )



# validation_steps = math.ceil( 28 / batch_size ) # 122 3D validation images  #  math.ceil((1023+987+541)/batch_size) # 250 # number of valid images/batch_size, round up
# steps_per_epoch = math.ceil( 67 / batch_size ) # 365 3D training images # math.ceil((2755+3424+1949)/batch_size) # 1000 # number of train imges/batch_size, round up
# 122 and 365 are total images.
# So we have 67 train images, and 28 valid images


initial_epoch = 0 # SHOULD THIE BE 1?


seed = 1
numFramesPerStack = 60
# NUM FRAMES PER STACK IS THE TOTAL NUMBER OF SLICES 


input_shape = ( 1, 256 , 256 , numFramesPerStack ) # (1, 200, 200, 352)


# SO THE CHANNELS IS SET  TO ALWAYS FIRST OR ALWAYS LAST. IN THIS CASE, CHANNELS IS SET TO ALWAYS LAST
# SO THE TENSOR DIMENSIONS IN ORDER ARE IMG_HEIGHT , IMG_WIDTH , NUM_SLICES, NUM_CHANNELS

# NOTE: THE FIRST TWO DIMENSIONS OF INPUT SHAPE CAN BE CHANGED TO ARBITRARY NUMBERS.
#       THE CODE AUTOMATICALLY INTERPOLATES TO THAT NUMBER. 
#           HOWEVER, THE NUM_FRAMES_PER_STACK MUST NOT BE CHANGED

img_height = input_shape[ 1 ]
img_width = input_shape[ 2 ]
nChannels = input_shape[ 0 ]



num_classes = 1 # NOTE: FOR SEGMENTATION, SEEMS UNNECESSARY. USED ONLY FOR CLASSIFIER


# batch size*iterations = number of images





###############################################################################
###############################################################################
###############            TRAIN IMAGES    ####################################
###############################################################################
###############################################################################



# Is mask is prob a boolean
# Changes interpolation, ONLY NEAREST NEIGHBOR INTERP FOR MASKS
#           Sets interpolation order to 0
#           also rounds values, if above 127 it becomes 255, if less it becomes 0
# Also changes the SliceImg to SliceImg / 255



fileIDList_train_image, fileIDToPath_train_image, fileIDToLabel_train_image = getFileMappingForDir(dirPath_train_image, numFramesPerStack)
fileIDList_length_train_image = len(fileIDList_train_image)


train_image_generator = Generator3D( fileIDList_train_image, fileIDToPath_train_image, 
    numFramesPerStack=numFramesPerStack, 
    batchSize = batch_size, 
    dim = ( img_height , img_width ) , nChannels = nChannels ,
    seed = seed ,  shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, 
    widthShiftRange=0, heightShiftRange=0, 
    flipLR = False, flipUD = False )



###############################################################################
###############################################################################
###############     VALIDATION IMAGES    ######################################
###############################################################################
###############################################################################



fileIDList_valid_image, fileIDToPath_valid_image, fileIDToLabel_valid_image = getFileMappingForDir(dirPath_valid_image, numFramesPerStack)
fileIDList_length_valid_image = len(fileIDList_valid_image)


valid_image_generator = Generator3D( fileIDList_valid_image , fileIDToPath_valid_image , 
    numFramesPerStack=numFramesPerStack, 
    batchSize = batch_size, 
    dim = ( img_height , img_width ) , nChannels = nChannels ,
    seed = seed , shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, 
    widthShiftRange=0, heightShiftRange=0, 
    flipLR = False, flipUD = False )



###   BATCH_SIZE , NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH , NUM_FRAMES_PER_STACK


# train_generator = zip(train_image_generator,train_mask_generator)

# valid_generator = zip(valid_image_generator,valid_mask_generator)

# def combine_generator( gen1 , gen2, total_num_images, batch_size ):
#     # Attempting to combine the generators instead of zipping them
    
#     index=random.randint(0,total_num_images-batch_size)
#     # index = 0
#     print('COMB GEN CALLED')
#     print('COMB GEN CALLED')
#     print(index)
#     print(index)

#     # index=0
#     print(np.shape(gen1.__getitem__(index)_))
#     print(np.shape( ( gen1.__getitem__(index) , gen2.__getitem__(index)  ) ))
#     while True:
#         yield ( gen1.__getitem__(index) , gen2.__getitem__(index)  )


def combine_generator( gen, total_num_images, batch_size ):

    counter = 0

    num_iterations = np.ceil( total_num_images / batch_size ).astype( int )

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


train_generator = combine_generator( train_image_generator, total_train_images,
    batch_size )

valid_generator = combine_generator( valid_image_generator ,total_valid_images,
    batch_size )




###############################################################################
###############################################################################
###############     Helper Functions  ######################################
###############################################################################
###############################################################################

smooth = 1

def dice_coef(y_true, y_pred):
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



# config = tensorflow.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
config = tensorflow.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)



tensorflow.keras.backend.set_session(tensorflow.Session(config=config))


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

# This section simply creates the output folder where the csv writer will save model info at
# Define folder name with model parameters
# will have type /3D_SPGR_Segmentation/out_b30_e300_se60_vs_100

output_folder = "b{}_e{}_se_{}_vs_{}".format(str(batch_size),str(epochs),
                                       str(steps_per_epoch),str(validation_steps))

outputDirPath = os.getcwd()+'/'+study_name+'/'+output_folder

if not os.path.exists(outputDirPath):
    os.makedirs(outputDirPath)
else:
    print('directory exists!!!!')

tensorboardDirPath = os.getcwd()+'/'+study_name+'/TENSORBOARD/'+output_folder
print(tensorboardDirPath)

if not os.path.exists(tensorboardDirPath):
    os.makedirs(tensorboardDirPath)
else:
    print('directory exists!!!!')


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
# NOTE: THIS IS JUST SAYING, GIVE AN INPUT SHAPE, AND SPECIFIC NETWORK ARCHITECTURE, 
#       DEFINE THE INITIAL BIASES AND WEIGHTS.








# MAtt mentioned this for prediction instead of getsimple3dunet
# model = tensorflow.keras.models.load_model(modelFilePath, custom_objets={"dice_coef": dice_coef})

model.compile( optimizer=Adam(2e-5), loss=tensorflow.keras.losses.binary_crossentropy, 
    metrics=[dice_coef,dice_coef_loss] )



#  DEFINES WHICH METRICS WILL BE USED TO ANALYZE THE MODELS EFFICACY

hist=model.fit_generator(generator = train_generator, validation_data=valid_generator, validation_steps= validation_steps, steps_per_epoch= steps_per_epoch, 
            epochs= epochs, verbose=1 ,  callbacks=callbackList , use_multiprocessing = 1,
            max_queue_size = 2)  


# generator = train_generator, defines the main data generator
# validation_data=valid_generator, defines the main validation generator
# Validation_steps
# steps_per_epoch = steps_per_epoch , total number of steps (batches of samples) to yield from generator before proceeding to next epoch
#           should be ceil( total_num_samples / batch_size )
#           thus it will cycle through entire data set one per epoch
# 
#  epochs is total number of epochs to train
# verbose = 1, 0 is silent, 1 prints progress bar, 2 prints one line per epoch
# max_queue_size = 10, this is the number of queues that the CPU will create of images while GPU is running
# callbacks = callbackList