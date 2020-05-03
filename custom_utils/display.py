import cv2
import numpy as np
import sys

sys.path.insert(0, "/d1/DeepLearning/custom_utils/")

import img_utils
import keyboard

windowIndex = 0

windowTitles = []

def displayStack(stack, title=""):
	if stack.dtype == np.dtype('bool'):
		stack = stack.astype('uint8') * 255
	title = getUniqueTitle(title)
	sliceIndex = 0
	while True:
		imgSlice = stack[:,:,sliceIndex]
		displayGrayImage(imgSlice, title=title, forceUniqueTitle=False)
		key = cv2.waitKeyEx()
		if key == 2490368: # UP
			sliceIndex += 1
			if sliceIndex >= stack.shape[2]:
				sliceIndex = 0
		elif key == 2621440: # DOWN
			sliceIndex -= 1
			if sliceIndex < 0:
				sliceIndex = stack.shape[2] - 1
		elif key == 27: # ESC
			cv2.destroyAllWindows()
			break


					
def displayGrayImage(image, title="", forceUniqueTitle=True):
	displayBGRImage(image, title, forceUniqueTitle=forceUniqueTitle)

def displayHSVImage(image, title="", forceUniqueTitle=True):
	# imshow expects images to be in BGR format
	image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
	displayBGRImage(image, title, forceUniqueTitle=forceUniqueTitle)

def getUniqueTitle(title):
	if not title:
		global windowIndex
		title = "Display: " + str(windowIndex)
		windowIndex = windowIndex + 1
	global windowTitles
	# Ensure title is unique
	if title in windowTitles:
		index = 0
		while True:
			if (title + str(index)) not in windowTitles:
				title = (title + str(index))
				break
			index += 1
	# Add new title to list
	windowTitles.append(title)
	return title
# Wrapper on cv2.imshow that makes window sizes a lot more convenient
def displayBGRImage(image, title="", forceUniqueTitle=True):
	if forceUniqueTitle:
		title = getUniqueTitle(title)
	
	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.moveWindow(title, 10,10)
	imDims = image.shape
	rWidth = imDims[0]
	cWidth = imDims[1]
	maxDisplayHeight = 700
	maxDisplayWidth = 1000

	if rWidth > cWidth:
		displayHeight = maxDisplayHeight if maxDisplayHeight < rWidth else rWidth
		displayWidth = int(cWidth/rWidth*displayHeight)
	else:
		displayWidth = maxDisplayWidth if maxDisplayWidth < cWidth else cWidth
		displayHeight = int(rWidth/cWidth*displayWidth)
	#cv2.resizeWindow(title, displayWidth,displayHeight)
	cv2.resizeWindow(title, int(800*cWidth/rWidth),800)
	cv2.imshow(title, image)

def waitForEscapeEvent():
	global windowTitles
	while True:
		k = cv2.waitKey(200)
		if k == 27: # ESC key (see Dec column of ASCII Table - http://www.asciitable.com/)
			cv2.destroyAllWindows()
		# Delete last displayed image
		elif k == 97: # 'a'
			title = windowTitles.pop(len(windowTitles)-1)
			cv2.destroyWindow(title)
		for title in windowTitles:
			if cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN ) < 0:
				cv2.destroyWindow(title)
				windowTitles.remove(title)
		if len(windowTitles) == 0:
			break

# http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
# Takes an RGB image and displays a green overlay with alpha value provided (default of 0.5)
def getBGRWithOverlay(image, mask, alpha=0.5, color=(0,255,0)):
	overlay = image.copy()
	output = image.copy()
	overlay[:,:,0][mask!=0] = color[0]
	overlay[:,:,1][mask!=0] = color[1]
	overlay[:,:,2][mask!=0] = color[2]
	cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
	return output
def getGrayWithOverlay(image, mask, alpha=0.5, color=(0,255,0)):
	image = grayToBGR(image)
	return getBGRWithOverlay(image, mask, alpha=alpha, color=color)

def grayToBGR(gray):
	grayDims = gray.shape
	grayBGR = np.zeros((grayDims[0], grayDims[1], 3), np.uint8)
	grayBGR[:,:,0] = gray
	grayBGR[:,:,1] = gray
	grayBGR[:,:,2] = gray
	return grayBGR

# Change image pixels, defined in mask, to green
def getBGRWithOutline(image, mask, alpha=0.5, color=(0,255,0)):
	mask = mask.copy()
	perimMask = img_utils.bwperim(mask)
	return getBGRWithOverlay(image, perimMask, alpha=alpha, color=color)

def getGrayWithOutline(image, mask, alpha=0.5, color=(0,255,0)):
	image = grayToBGR(image)
	return getBGRWithOutline(image, mask, alpha=alpha, color=color)