
from custom_utils import csv_utils, old_custom_generators
from custom_utils.old_custom_generators import getFolderNamesFromDir, getFileNamesFromDir, getFileFrameNumber, getFileMappingForDir
# from custom_utils.custom_generators import Generator3DClassifier
from custom_utils.custom_generator_3D_PADTO64 import Generator_3D_PADTO64
# from Generator3DClassifier import getVolumeForFileID
# import nibabel as nib
import string
import os
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization 
from keras import optimizers
import csv
import custom_utils.custom_models

#import custom_models
import time
import random
import math
import PIL
import scipy.io


def combine_generator( gen, total_num_images, batch_size ):
    # Attempting to combine the generators instead of zipping them
    
    index=random.randint(0,total_num_images-batch_size)



    img , mask = gen.__getitem__(index)

    while True:
        yield ( img , mask )


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    dice_coef_loss_out = 1 - dice_coef( y_true , y_pred )
    return dice_coef_loss_out 



# check out uptake script (end) and Matt's script to see how to load .h5 weights and evaluate model on test data 
# load weights epoch 13
# load test data: images and labels, convert label to 1-hot (categorical), need to build stack 
# evaluate model, percent accurate
# csv file of each case and whether it was correct prediction or not
# 

# need to load test data and labels like load into generator, but need to extract info for csv file


# fileIDToLabel, i want the label from this variable, save to/append to testLabels variable, have to skip every 47 
# append iteration number from stack to stack
# generate volume and data with a batch size equal to all the testdata (length of image files/47)
# use data generator without augmentation, and output the 3d images (batch=all) and labels

# dirPath_test = 'Y:/data/NickJ/PET Deep Learning/For 3D Network/Test/'




batch_size = 67
batch_size_valid = 95 - batch_size
seed = 1
numFramesPerStack = 60
total_train_images = 67
total_valid_images = 65 - total_train_images
# NUM FRAMES PER STACK IS THE TOTAL NUMBER OF SLICES 
pad_to_64 = True
zoomRange= (1,1) # around 1
rotationRange=0 # in degrees
horz_shift = 0 # % of the total number of pixels
vert_shift = 0
flipLR = False
flipUD = False
bool_shuffle = False
batch_size = 20



# # dirPath_test = 'Y:/lab/njosselyn/PET Deep Learning/repeat/Train/'
# dirPath_test = '/d1/DeepLearning/3D_data/3D_data/train/'

dirPath_valid = "/d1/DeepLearning/3D_data/3D_data/valid/"

dirPath_valid_mask = (dirPath_valid + "mask/")
dirPath_valid_image = (dirPath_valid + "image/")



# numFramesPerStack = 47
numFramesPerStack = 60

# fileIDList, fileIDToPath, fileIDToLabel = getFileMappingForDir(dirPath_test, numFramesPerStack)
# Grabs strings of file names


fileIDList_valid_image, fileIDToPath_valid_image, fileIDToLabel_valid_image = getFileMappingForDir(dirPath_valid_image, numFramesPerStack)
fileIDList_length_valid_image = len(fileIDList_valid_image)


# batch_size = len(fileIDList) # number of 3D image volumes 
label_list = []
id_list = []

# for idlabel in fileIDList:
# 	label_list.append(fileIDToLabel[idlabel])
# 	id_list.append(idlabel[4:8])

# print('fileIDList')
# print(fileIDList)
# print('label_list')
# print(label_list)
# print('batch_size')
# print(batch_size)
# print('id_list')
# print(id_list)
# output the label_list and id_list to csv columns

img_input_size = 128
input_shape = ( 1, img_input_size , img_input_size , 64 ) # (1, 200, 200, 352)

# input_shape = (100, 100, numFramesPerStack, 1) # (1, 200, 200, 352)

nChannels = input_shape[0]
num_classes = 1

dim = (input_shape[1],input_shape[2]) # x,y dimensions of a 2D slice

seed = 1
septoken = "_"

# # Augmentation parameters (all off)
# shuffleTF = False
# zoomrange = (1, 1) # (1,1)
# rotationrange = 0
# widthshiftrange = 0
# heightshiftrange = 0
# fliplr = False # False
# flipud = False
# customlength = -1
# # balanceclasses = False # try unbalanced classes too (False) 


valid_image_generator = Generator_3D_PADTO64( fileIDList_valid_image , fileIDToPath_valid_image , 
        numFramesPerStack=numFramesPerStack, 
        batchSize = batch_size, 
        dim = ( img_input_size , img_input_size ) , nChannels = nChannels ,
        seed = seed , shuffle=bool_shuffle, sepToken="_", zoomRange=zoomRange, rotationRange=rotationRange, 
        widthShiftRange=vert_shift, heightShiftRange=horz_shift, 
        flipLR = flipLR, flipUD = flipUD )



test_generator = combine_generator( valid_image_generator, total_valid_images,
    batch_size_valid )



# test_generator = Generator3DClassifier(fileIDList, fileIDToPath, fileIDToLabel, numFramesPerStack,
#     seed=seed, batchSize=batch_size, dim=dim, nChannels=nChannels,
#     nClasses=num_classes, shuffle=shuffleTF, sepToken=septoken, zoomRange=zoomrange, rotationRange=rotationrange, widthShiftRange=widthshiftrange, 
#     heightShiftRange=heightshiftrange, flipLR = fliplr, flipUD = flipud, customLength=customlength, balanceClasses=balanceclasses)

# fileIDToLabel[fileIDList] # loop over fileIDList list and use that to loop through the fileIDToLabel dictionary to access the 
    # labels, need to skip every 47 to then get the next patient ID and output the label to csv
# build 3D volumes and labels using generator without data augmentation and batch size equal to all images, save as testdata and testlabels


gentype=type(test_generator)

print('Generator Output Data Type')
print(gentype)


print('Evaluating Model')
# model = keras.models.load_model('Y:/data/NickJ/PET Deep Learning/code/predictions_1016/weights_10_15_19_AUG.13-0.70-0.71-0.70-0.70.h5')

# model_file_path = '/d1/DeepLearning/3D_SPGR_Segmentation_deepresultsJan9/b1_e99_se_71.0_vs_24.0/'
model_file_path = '/d1/DeepLearning/3D_SPGR_Segmentation_deepresultsJan9/b1_e99_se_71.0_vs_24.0/'
model_file_name = '3D_SPGR_Segmentation.70-0.08-0.66.h5'
model_file_total = model_file_path + model_file_name

print('MODEL FILE NAME')
print(model_file_total)
print()
print()



model = keras.models.load_model( model_file_total )


# model = keras.models.load_model('Y:/lab/njosselyn/PET Deep Learning/For_3D_Network_histeq/predictions/weights_11_12_19_AUG_histeq.06-0.66-0.72-0.71-0.64.h5')

model.summary()

# j=dir(test_generator)
# print(j)

# We want output from the 3D Generator data generation

# Q = test_generator._Generator3DClassifier__data_generation(fileIDList)
Q = test_generator._Generator_3D_PADTO64__data_generation(fileIDList)


# WE WANT THE INPUT TO EVALUATE FUNCTION TO BE A TUPLE
# SO ( IMAGE , MASK )
# CAN TRY .GET_ITEM__

qtype=type(Q)
print('Data Type of Q')
print(qtype)
q0=Q[0] # img
q1=Q[1] # mask

# print('q0')
# print(q0)
# print('q1')
# print(q1)



# model.evaluate actually does the prediction
# going from custom generator to evaluating data types arent necessarily consistent
# 




# Evaluate model loss and accuracy
score = model.evaluate(Q[0],Q[1], batch_size=20) 


print("Finished evaluating")
print("Loss of %0.4f" % score[0])
print("Accuracy of %0.4f" % score[1])



# # prediction results, percent possible for each label category
# predictions = model.predict(Q[0], batch_size=20)

# print("Finished predicting")
# print("Predictions")
# print(predictions)

# # predicted class/label
# class_predictions = model.predict_classes(Q[0], batch_size=20)
# print("Finished Predicting Classes")
# print("Class Predictions")
# print(class_predictions)
# # model.predict() or model.predict_generator() 

# ## max percents in class_predictions list
# max_pred = []
# for plabel in range(0,len(predictions)): # plabel = "patient label"
# 	max_pred.append(max(predictions[plabel]))

# pred0 = []
# pred1 = []
# pred2 = []
# for prob in range(0,num_classes):
# 	for pat in range(0,len(predictions)):
# 		if prob == 0:
# 			pred0.append(predictions[pat][prob])
# 		elif prob == 1:
# 			pred1.append(predictions[pat][prob])
# 		elif prob == 2:
# 			pred2.append(predictions[pat][prob])


# # load FDG IDs for test data from .mat file
# # test_FDG_IDs_matfile = scipy.io.loadmat("Y:/data/NickJ/PET Deep Learning/For 3D Network/test_FDG_IDs.mat")
# test_FDG_IDs_matfile = scipy.io.loadmat("Y:/lab/njosselyn/PET Deep Learning/repeat/FDG_ID_seg_predict.mat")
# test_FDG_IDs = test_FDG_IDs_matfile['FDG_ID'][0]


# ## Confusion Matrix ##
# import sklearn.metrics
# import seaborn as sn
# import pandas as pd
# #matplotlib inline
# import matplotlib.pyplot as plt
# actual = to_categorical(label_list,3)
# predicted_cm = np.argmax(predictions, axis=1)
# actual_cm = np.argmax(actual,axis=1)
# matrix = sklearn.metrics.confusion_matrix(actual_cm, predicted_cm)
# # Display absolute valued confusion matrix
# df_cm = pd.DataFrame(matrix, range(3),
#                   range(3))
# plt.figure(figsize=(12,10))
# sn.set(font_scale=1)#for label size
# ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cmap="Blues", fmt='g')# font size
# ax.set(xlabel='Predicted', ylabel='Actual', title="Absolute Values for Predicted vs. Actual")

# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values

# plt.show()

# # Display relative-value confusion matrix
# matrixRel = []
# for row in matrix:
#     row = row / np.sum(row)
#     matrixRel.append(row)
    
# df_cm = pd.DataFrame(matrixRel, range(3),
#                   range(3))
# plt.figure(figsize=(12,10))
# sn.set(font_scale=1)#for label size
# ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cmap="Blues", fmt='0.3f')# font size
# ax.set(xlabel='Predicted', ylabel='Actual', title="Relative Values for Predicted vs. Actual")

# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values

# plt.show()

# # Classification report
# class_report = sklearn.metrics.classification_report(actual_cm, predicted_cm)
# print('Classification Report:')
# print(class_report)
# acc_report = sklearn.metrics.accuracy_score(actual_cm,predicted_cm)
# print('Accuracy Report:')
# print(acc_report)



# #### Writing to CSV files ####

# # x = list(range(1,123)) # temporary variable until incorporate FDG_ID labels (for test data) from MATLAB
# header = ['FDG_ID', 'Deidentified ID', 'Ground Truth Label', 'Predicted Label', 'Percent Certain', 'Percent per possible category', 'Prob0', 'Prob1', 'Prob2']
# with open('split_all_probs_histeq_epoch6.csv','w') as pred:
# 	writer = csv.DictWriter(pred, lineterminator='\n', fieldnames=header)
# 	writer.writeheader()
# 	for i in range(0,len(test_FDG_IDs)):
# 		writer.writerow({"FDG_ID": test_FDG_IDs[i], "Deidentified ID": id_list[i], "Ground Truth Label": label_list[i], "Predicted Label": class_predictions[i], "Percent Certain": max_pred[i], "Percent per possible category": predictions[i], 'Prob0': pred0[i], 'Prob1': pred1[i], 'Prob2': pred2[i]})
	
# header2 = ['Percent Certain', 'predicted right definitely', 'predicted right poorly', 'predicted right averagely', 'predicted wrong definitely/badly', 'predicted wrong barely', 'predicted wrong averagely poor']
# with open('thresholds_histeq_epoch6.csv','w') as pred2:
# 	writer2 = csv.DictWriter(pred2, lineterminator='\n', fieldnames=header2)
# 	writer2.writeheader()
# 	for j in range(0,len(test_FDG_IDs)): 
# 		# predicted right
# 		if max_pred[j]>0.85 and class_predictions[j]==label_list[j]:
# 			writer2.writerow({'predicted right definitely': test_FDG_IDs[j]})
# 		elif max_pred[j]<0.50 and class_predictions[j]==label_list[j]:
# 			writer2.writerow({'predicted right poorly': test_FDG_IDs[j]})
# 		elif max_pred[j]<0.85 and max_pred[j]>0.50 and class_predictions[j]==label_list[j]:
# 			writer2.writerow({'predicted right averagely': test_FDG_IDs[j]})
# 		# predicted wrong
# 		elif max_pred[j]>0.85 and class_predictions[j]!=label_list[j]:
# 			writer2.writerow({'predicted wrong definitely/badly': test_FDG_IDs[j]})
# 		elif max_pred[j]<0.50 and class_predictions[j]!=label_list[j]: 
# 			writer2.writerow({'predicted wrong barely': test_FDG_IDs[j]})
# 		elif max_pred[j]<0.85 and max_pred[j]>0.50 and class_predictions[j]!=label_list[j]: 
# 			writer2.writerow({'predicted wrong averagely poor': test_FDG_IDs[j]})

# # make new for loop for if/elif statements and write to a second .csv file
# # output train loss, accuracy? 

