import cv2
import numpy as np
from skimage.measure import label

def bwperim(mask):
	kernel = np.ones((3,3), np.uint8)
	maskEroded = cv2.erode(mask, kernel, iterations=1)
	return (mask - maskEroded)

def imfill_stack(stack):
	for sliceIndex in range(0, stack.shape[2]):
		imgSlice = stack[:,:,sliceIndex]
		imgSlice, hadHoles = imfill(imgSlice)
		stack[:,:,sliceIndex] = imgSlice
	return stack

def imfill(im_th):
	if im_th.dtype == np.dtype('bool'):
		returnBool = True
		im_th = im_th.astype('uint8') * 255
	else:
		returnBool = False

	im_th = np.pad(im_th, pad_width=1, mode='constant', constant_values=(0))
	#im_th = np.squeeze(im_th)
	# Copy the thresholded image.
	im_floodfill = im_th.copy()
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_th.shape[:2]
	mask = np.zeros((h+2, w+2, 1), np.uint8)
	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);
	 
	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	 
	# Combine the two images to get the foreground.
	im_out = im_th | im_floodfill_inv
	#pauseAndDisplay(im_floodfill_inv)

	if np.any(im_floodfill_inv > 0):
		hadHoles = True
	else:
		hadHoles = False

	im_out = im_out[1:(im_out.shape[0]-1), 1:(im_out.shape[1]-1)]
	if returnBool:
		return im_out.astype('bool'), hadHoles
	else:
		return im_out, hadHoles

def getLargestCC(mask, numObjects=1):
	labels = label(mask)
	binCounts = np.bincount(labels.flat)
	binCounts[0] = 0
	#print(np.argmax(binCounts))
	indsDescend = np.argsort(-binCounts)
	indsToInclude = indsDescend[0:numObjects]
	#print(indsToInclude)
	mask = np.full(labels.shape, False)
	for ind in indsToInclude:
		mask[labels == ind] = True
	return mask
	#largestCC = labels == np.argmax(binCounts)
	#return largestCC