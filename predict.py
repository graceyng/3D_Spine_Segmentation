import tensorflow
import numpy as np
import string
import os
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import csv
import custom_models
import time
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras import backend as K
from scipy.misc import imread, imresize, imsave
import cv2
from matplotlib import pyplot as plt
import glob
from pathlib import Path
from skimage import img_as_ubyte, measure
import time

#import display

smooth = 1

img_size = 128



def dice_coef(y_true, y_pred):
	# y_true = float(y_true)
	# y_pred = float(y_pred)
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_manual(img1,img2):
	
	img1[img1>0.5]=1
	img1[img1<=0.5]=0
	img2[img2>0.5]=1
	img2[img2<=0.5]=0
	img1=img_as_ubyte(img1)
	img2=img_as_ubyte(img2)

	intersection = np.logical_and(img1,img2)
	# pos_intersection = np.logical_and(img1 != 0,img2 != 0)
	# neg_intersection = np.logical_and(np.invert(img1), np.invert(img2))

	# intersection = pos_intersection+neg_intersection
	# intersection[intersection>0]=255

	img1[img1>0.5]=1
	img1[img1<=0.5]=0
	img2[img2>0.5]=1
	img2[img2<=0.5]=0

	# print(str(intersection.sum()))
	# print(str(256*256))

	# print(str(img1.shape))
	# print(str(img2.shape))

	return round(2.*intersection.sum() / ( img1.sum() + img2.sum() ),2)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)   








modelFilePath = "/d1/DeepLearning/output_20_batch/hip_model.52-2.87-0.91.h5"
model = tensorflow.keras.models.load_model(modelFilePath, custom_objects={"dice_coef": dice_coef})



image_files_to_segment = glob.glob("/d1/DeepLearning/data/valid/image/dummy_class/*.png")
# Grabs all validation image file names
# Now I need to load them one at a time
num_images = len(image_files_to_segment)

for cc in range( num_images):
	


	#imgFilePath = "/d1/DeepLearning/data/valid/image/dummy_class/12019.png"
	Full_img_File_Path = image_files_to_segment[cc]

	img = imread( Full_img_File_Path )

	img = cv2.resize(img, ( img_size , img_size ), interpolation=cv2.INTER_CUBIC)

	# normalize again after interpolation
	if np.amax(img)!=255:
		img=img/np.amax(img)*255

	img_File_Path = Path ( Full_img_File_Path )
	img_File_Name = img_File_Path.name



	for iter  in range(len(img_File_Path.parts)):
		# print("Length is "+str(len(img_File_Path.parts)-2))
		# print("Iter is: "+str(iter))
		# print(img_File_Path.parts[iter])
		if iter == len(img_File_Path.parts)-3:
			mask_file_name = mask_file_name+'/mask'
			continue

		if iter == 0:
			mask_file_name = img_File_Path.parts[iter]
		elif iter==1:
			mask_file_name = mask_file_name+img_File_Path.parts[iter]
		else:
			mask_file_name = mask_file_name+'/'+img_File_Path.parts[iter]

		if iter==len(img_File_Path.parts)-4:
			save_path = mask_file_name


	# print(mask_file_name)
	manual_mask = imresize( imread(mask_file_name) , ( img_size , img_size ) )

	if np.amax(manual_mask)!=255:
		manual_mask=manual_mask/np.amax(manual_mask)*255


	# print("img shape after vs2 resize is: "+str(img.shape))
	img = img[ np.newaxis , : , : , np.newaxis ]
	# img axes ar in order nImages , nRows, ncols, Z axis or RGB
	# if img were 3D z would be used

	# mask is probability map between 0 and 1
	mask = model.predict(img)
	
	mask_heatmap = mask

	mask_heatmap = mask_heatmap/np.amax(mask_heatmap)

	mask_heatmap = np.squeeze(mask_heatmap)


	# Now turn 4D matrix back into 2D image
	mask = np.squeeze(mask)
	img = np.squeeze(img)


	mask[mask > 0.5] = 255   # Binarize the mask file
	mask[mask <= 0.5] = 0


	fig1 = plt.figure()
	ax1 = plt.subplot(1,5,1)
	ax1.imshow(img,'gray',interpolation='none')
	ax1.set_title("Image "+img_File_Name )
	
	
	ax2 = plt.subplot(1,5,2)
	ax2.imshow(manual_mask,cmap='gray',interpolation='none')
	ax2.set_title("Dice similarity coefficient: "+str(dice_coef_manual(mask,manual_mask))) 


	ax3 = plt.subplot(1,5,3)
	ax3.imshow(mask,cmap='gray',interpolation='none')

	ax4 = plt.subplot(1,5,4)
	ax4.imshow(mask_heatmap,'jet',interpolation='none')
	ax4.set_title('Heatmap')

	ax5 = plt.subplot(1,5,5)
	ax5.imshow(img,'gray',interpolation='none')
	ax5.imshow(mask,'jet',interpolation='none',alpha=0.5)
	ax5.set_title('Overlayed')


	new_dir = save_path+"/predict_{}/".format(time.strftime("%Y_%m_%d"))
	fig1.savefig(new_dir+"Results_{}".format(img_File_Name))
	# plt.show()

	
	# plt.figure()
	# plt.subplot(1,5,1)
	# plt.imshow(img,'gray',interpolation='none')
	# plt.title("Image "+img_File_Name ,loc='Center')
	

	# plt.subplot(1,5,2)
	# plt.imshow(manual_mask,'gray',interpolation='none')
	# plt.title("Dice similarity coefficient: "+str(dice_coef_manual(mask,manual_mask))) #+dice_coef(mask,img))

	# plt.subplot(1,5,3)
	# plt.imshow(mask,'gray',interpolation='none')

	# plt.subplot(1,5,4)
	# plt.imshow(mask_heatmap,'jet',interpolation='none')
	# plt.title('Probability Heatmap for Mask')


	# plt.subplot(1,5,5)
	# plt.imshow(img,'gray',interpolation='none')
	# plt.imshow(mask,'jet',interpolation='none',alpha=0.5)
	# plt.title('Overlayed')




	# new_dir = save_path+"/predict_{}/".format(time.strftime("%Y_%m_%d"))
	# plt.savefig(new_dir+"Results_{}.png".format(img_File_Name))
	# # plt.show()

	print(" New directory is: ")

	try:
		os.mkdir(new_dir)
	except:
		print("directory "+new_dir+" already exists!")


	save_name = img_File_Name[0:len(img_File_Name)-4]+"_MASKED.png"
	print(new_dir+save_name)
	print(save_name)
	print(new_dir)

	plt.imsave( new_dir+save_name , mask, cmap='gray', format='png')


	plt.close('all')


