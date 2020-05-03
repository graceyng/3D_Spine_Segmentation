
from keras.utils import Sequence
import numpy as np 
import glob
import os
import keras
import gc
# import PIL
# from PIL import Image
# import Image
import scipy
import cv2
#from tifffile import imread
# from skimage import skimage.transform 
import scipy.misc
# from matplotlib import pyplot as plt

###############################################################################
###############################################################################
############### GEN DIRECTORY FUNCTIONS    ####################################
###############################################################################
###############################################################################

def getFolderNamesFromDir(dirPath, namesOnly=True):

	folderPaths = glob.glob(os.path.join(os.path.normpath(dirPath), "*") + "/")
	if not namesOnly:
		return folderPaths
	folderNames = sorted([os.path.basename(os.path.normpath(path)) for path in folderPaths])
	return folderNames



def getFileNamesFromDir(dirPath, namesOnly=True):
	filePathList = glob.glob(os.path.join(os.path.normpath(dirPath), "*"))
	for filePath in filePathList:
		if not os.path.isfile(filePath):
			raise ValueError("%s is not a file" % filePath)
	if not namesOnly:
		return filePathList
	fileNameList = sorted([os.path.basename(os.path.normpath(path)) for path in filePathList])
	return fileNameList




def getFileFrameNumber(fileName, sepToken="_"):
	fileName, _ = os.path.splitext(fileName)
	splitIndex = fileName.rfind(sepToken)
	fileNameBase = fileName[0:(splitIndex)]
	frameNumStr = fileName[(splitIndex+1):]
	try:
		frameNum = int(frameNumStr)
	except ValueError:
		raise ValueError("%s does not contain properly formatted frame number: %s" % (fileName, frameNumStr))
	return (frameNum, fileNameBase)




def getFileMappingForDir(dirPath, numFramesPerStack, sepToken="_"):
	folderNameList = getFolderNamesFromDir(dirPath)
	fileIDToPath = {}
	fileIDToLabel = {}
	fileIDList = []
	numClasses = len(folderNameList)


	for folderIndex, folderName in enumerate(folderNameList):
		print("Extracting filepaths for: %s" % folderName)
		fileNameList = getFileNamesFromDir(os.path.join(dirPath, folderName))
		
		videoFileIDList = []
		videoFilePathList = []
		
		lastFileFrameNumber = None
		for fileIndex, fileName in enumerate(fileNameList):
			fileNameNoExt, _ = os.path.splitext(fileName)
			fileID = "%03d_%s" % (folderIndex, fileNameNoExt)
			if fileName == 'Thumbs.db':
				dummy = ['Thumbs.db file found, ignore']

			else:
				dummy = ['No Thumbs.db']
				fileFrameNumber, fileNameBase = getFileFrameNumber(fileName, sepToken=sepToken)

			if lastFileFrameNumber != None and fileFrameNumber != (lastFileFrameNumber + 1):
				if len(videoFileIDList) < numFramesPerStack:
					print("""Number of frames in video is less than number requested per stack with %d frames:\n Video will not be included in dataset\nFirst file of video: %s """ % (len(videoFileIDList), videoFilePathList[0]))
					videoFileIDList.clear()
					videoFilePathList.clear()
				else:
					# Add all elements to dictionary but subtract out first numFramesPerStack-1 before adding to fileIDList
					#videoFilePathList = videoFilePathList[0:(len(videoFilePathList)-numFramesPerStack+1)]
					for videoFileIndex in range(0, len(videoFileIDList)):
						fileIDToPath[videoFileIDList[videoFileIndex]] = videoFilePathList[videoFileIndex]
						fileIDToLabel[videoFileIDList[videoFileIndex]] = folderIndex
					videoFileIDList = videoFileIDList[0:(len(videoFileIDList)-numFramesPerStack+1)]
					fileIDList += videoFileIDList
					videoFileIDList.clear()
					videoFilePathList.clear()
			lastFileFrameNumber = fileFrameNumber
			videoFileIDList.append(fileID)
			videoFilePathList.append(os.path.join(dirPath, folderName, fileName))

		# Remove last numFramesPersStack-1 from each video series and then add to dictionary
		videoFileIDList = videoFileIDList[0:(len(videoFileIDList)-numFramesPerStack+1)]
		videoFilePathList = videoFilePathList[0:(len(videoFilePathList)-numFramesPerStack+1)]
		length_videoFileIDList = len(videoFileIDList)
		if dummy[0] == 'Thumbs.db file found, ignore':
			for videoFileIndex in range(0, length_videoFileIDList-1): # -1 to get rid of Thumbs file
				fileIDToPath[videoFileIDList[videoFileIndex]] = videoFilePathList[videoFileIndex]
				fileIDToLabel[videoFileIDList[videoFileIndex]] = folderName
			videoFileIDList.clear()
			videoFilePathList.clear()
		elif dummy[0] == 'No Thumbs.db':
			for videoFileIndex in range(0, length_videoFileIDList): 
				fileIDToPath[videoFileIDList[videoFileIndex]] = videoFilePathList[videoFileIndex]
				fileIDToLabel[videoFileIDList[videoFileIndex]] = folderName
			videoFileIDList.clear()
			videoFilePathList.clear()

	return fileIDList, fileIDToPath, fileIDToLabel




	#import view_classification_models as models


###############################################################################
###############################################################################
###############     GENERATOR FUNCTIONS    ####################################
###############################################################################
###############################################################################

# def combine_generator( gen1 , gen2 ):
# 	# Attempting to combine the generators instead of zipping them
# 	while True:
# 		yield ( gen1.next() , gen2.next() )





class Generator_3D(Sequence):


	def __init__(self, fileIDList, fileIDToPath, numFramesPerStack,  batchSize=32, dim=(224,224), 
				 nChannels=1, seed=1, shuffle=False, sepToken="_", zoomRange=(1,1), 
				 rotationRange=0, widthShiftRange=0, heightShiftRange=0,
				 flipLR=False, flipUD=False):

		'Initialization'


		self.fileIDList = fileIDList
		self.fileIDToPath = fileIDToPath

		self.dim = dim
		self.batchSize = batchSize
		self.nChannels = nChannels
		self.shuffle = shuffle
		self.random = np.random.RandomState(seed=seed)
		self.on_epoch_end()
		self.numFramesPerStack = numFramesPerStack
		self.sepToken = sepToken

		self.zoomRange = zoomRange
		self.rotationRange = rotationRange
		self.widthShiftRange = widthShiftRange
		self.heightShiftRange = -heightShiftRange

		self.flipLR = flipLR
		self.flipUD = flipUD


	def __getitem__(self, index):



		file_indices_to_grab = (np.arange(self.batchSize)+index).astype(int)

		batchFileIDList = np.array( self.fileIDList ) [ file_indices_to_grab ]
		

		Img , Mask = self.__data_generation(batchFileIDList)

		return Img , Mask


	def on_epoch_end(self):

		self.indexes = np.arange( len( self.fileIDList ) )


		if self.shuffle:

			self.random.shuffle(self.indexes)

	def __data_generation(self, fileIDBatch):
		# print('Calling data generation . . .')
		# print()

		while True: 
			# THIS WHILE TRUE IS CRUCIAL FOR THE KERAS TO RUN THE GENERATOR PROPERLY

			'Generates data containing batchSize samples' # X : (n_samples, *dim, nChannels)
			# Initialization
		
			Img = np.empty((self.batchSize, self.nChannels, *self.dim, self.numFramesPerStack), dtype=int)
			Mask = np.empty((self.batchSize, self.nChannels, *self.dim, self.numFramesPerStack), dtype=int)
				
			# Generate data
			for counter in range(self.batchSize):

				fileID = fileIDBatch[counter]

				Img[counter,:,:,:,:] , Mask[counter,:,:,:,:] = self.getVolumeForFileID(fileID)

			return Img , Mask



	def getVolumeForFileID(self, fileIDStart):


		#print(fileIDStart)
		"""
		Generates volume starting at given fileNameStart of height self.numFramesPerStack
		For example if fileNameStart is '00263523_054' (first part is MRN) and numFramesPerStack 
		is 5 then. The volume will be composed of images 00263523_054 through 00263523_058
		stacked on top of each other
		"""


		splitIndex = fileIDStart.rfind(self.sepToken)


		if splitIndex == -1:
			raise ValueError("Separation token not found for fileID: " % fileIDStart)

		fileIDBase = fileIDStart[0:(splitIndex)]
		
		frameNumStartStr = fileIDStart[(splitIndex+1):]
		

		numDigitsInNum = len(frameNumStartStr)

		Img_stack = np.empty((*self.dim, self.numFramesPerStack), dtype=np.uint8)
		Mask_stack = np.empty((*self.dim, self.numFramesPerStack), dtype=np.uint8)

		try:
			frameNumStart = int(frameNumStartStr)
		except ValueError:
			raise ValueError("Cannot cast %s to int for fileID %s" % (frameNumStartStr, fileIDStart))
		
		# Get the random augmentation states
		zf = self.getRandomZoomConfig(self.zoomRange)
		theta = self.getRandomRotation(self.rotationRange)
		tx, ty = self.getRandomShift(*self.dim, self.widthShiftRange, self.heightShiftRange)

		if self.flipLR:
			flipStackLR = self.getRandomFlipFlag()
		else:
			flipStackLR = False
		if self.flipUD:
			flipStackUD = self.getRandomFlipFlag()
		else:
			flipStackUD = False
		# Get each slice and apply augmentation


		for counter in range( self.numFramesPerStack ):
			
			fileNameNumStr = '{num:0{width}}'.format(num=counter+1, width=numDigitsInNum)
			fileID = "%s%s%s" % (fileIDBase, self.sepToken, fileNameNumStr)
			
			
			fileID_image =	self.fileIDToPath[fileID]

			fileID_mask = fileID_image.replace('image','mask')
			

			Img_slice = cv2.imread( fileID_image )
			Img_slice = cv2.cvtColor( Img_slice , cv2.COLOR_BGR2GRAY )
			Img_slice = np.asarray( Img_slice )

			Mask_slice = cv2.imread( fileID_mask )
			Mask_slice = cv2.cvtColor( Mask_slice , cv2.COLOR_BGR2GRAY )
			Mask_slice = np.asarray( Mask_slice )


			if np.amax(Mask_slice)==1:
				Mask_slice=Mask_slice*255
			


			Img_slice = cv2.resize( Img_slice , self.dim , interpolation=cv2.INTER_CUBIC ).astype(int)
			Mask_slice = cv2.resize( Mask_slice , self.dim , interpolation=cv2.INTER_CUBIC ).astype(int)
									
			Mask_slice = self.applyZoom( Mask_slice , zf , 1 )
			Img_slice = self.applyZoom( Img_slice , zf , 0 )
			
			Mask_slice = self.applyRotation( Mask_slice , theta , 1 )
			Img_slice = self.applyRotation( Img_slice , theta , 0 )
			
			Mask_slice = self.applyShift( Mask_slice , tx, ty , 1 )
			Img_slice = self.applyShift( Img_slice , tx, ty , 0 )


			if flipStackLR:

				Img_slice = np.fliplr( Img_slice )
				Mask_slice = np.fliplr( Mask_slice )

			if flipStackUD:

				Img_slice = np.flipup( Img_slice )
				Mask_slice = np.flipup( Mask_slice )

			if np.amax(Mask_slice)==255:
				Mask_slice = Mask_slice / 255


			Img_stack[ : , : , counter ] = Img_slice
			Mask_stack[ : , : , counter ] = Mask_slice
			

		return  Img_stack , Mask_stack


	def getRandomFlipFlag(self):


		return self.random.choice([True, False])



	def getRandomZoomConfig(self, zoomRange ):
		if zoomRange[0] == 1 and zoomRange[1] == 1:
			zf = 1
		else:
			zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
		return zf

	def applyZoom(self, img, zf, isMask , fill_mode='nearest', cval=0., interpolation_order=0):
		# BASED ON RANDOMLY DEFINED ZOOM VALUE FROM GETRANDOMZOOMCONFIG
		#
		#	APPLIES THE RANDOM ZOOM.
		#
		#	tHE SELF.ISMASK IS IMPORTANT PARAMETER BECAUSE IT DEFINES THE
		# INTERPOLATION TYPE. YOU ALWAYS WANT THE MASK TO REMAIN EITHER 0-1 OR 0-255 AND
		#		NOT BE ANY VALUES IN THE MIDDLE. HOWEVER, THE IMAGE CAN BE ANY UINT8
	

		if isMask:
			interp = cv2.INTER_NEAREST
			interpolation_order = 0
		else:
			interp = cv2.INTER_CUBIC
			interpolation_order = 1

		origShape = img.shape[1::-1]
		
		

		img = scipy.ndimage.zoom(img, zf, mode=fill_mode, cval=cval, order=interpolation_order)
		if zf < 1:
			canvas = np.zeros(origShape, dtype=img.dtype)
			rowOffset = int(np.floor((origShape[0] - img.shape[0])/2))
			colOffset = int(np.floor((origShape[1] - img.shape[1])/2))
			canvas[rowOffset:(rowOffset+img.shape[0]), colOffset:(colOffset+img.shape[1])] = img
			img = canvas
		elif zf > 1:
			rowOffset = int(np.floor((img.shape[0] - origShape[0])/2))
			colOffset = int(np.floor((img.shape[1] - origShape[1])/2))
			img = img[rowOffset:(rowOffset+origShape[0]), colOffset:(colOffset+origShape[1])]


		img = cv2.resize(img, origShape, interpolation=interp)
		


		return img


	
	def getRandomRotation(self, rotationRange):
		# RANDOMLY GENERATES THE ROTATION THAT WILL BE USED

		theta = self.random.uniform(-rotationRange, rotationRange)
		return theta



	def applyRotation(self, img, theta, isMask ):
		# APPLIES ROTATION ABOUT Z AXIS TO EACH IMAGE IN STACK

		"""Performs a random rotation of a Numpy image tensor.
		# Arguments
			x: Input tensor. Must be 3D.
			rg: Rotation range, in degrees.
			row_axis: Index of axis for rows in the input tensor.
			col_axis: Index of axis for columns in the input tensor.
			channel_axis: Index of axis for channels in the input tensor.
			fill_mode: Points outside the boundaries of the input
				are filled according to the given mode
				(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
			cval: Value used for points outside the boundaries
				of the input if `mode='constant'`.
			interpolation_order int: order of spline interpolation.
				see `ndimage.interpolation.affine_transform`
		# Returns
			Rotated Numpy image tensor.
		"""
		

		if isMask:

			origShape = img.shape

			img_out = scipy.ndimage.rotate( img , theta , reshape=False , order=3 , 
				mode='constant' , cval=0 )

			img_out [ img_out < 127 ] = 0
			img_out [ img_out >= 127] = 255
			img_out = img_out.astype(int)


		else:
			origShape = img.shape

			img_out = scipy.ndimage.rotate( img , theta , reshape=False , 
				order=3 , mode='constant' , cval=0 )

			img_out [ img_out < 0 ] = 0
			img_out [ img_out > 255] = 255
			img_out = img_out.astype(int)

		return img_out


	def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
		# RANDOMLY DEFINES TRANSLATION IN X AND Y DIRECTIONS

		tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
		ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w

		return (tx, ty)

	def applyShift(self, img, tx, ty, isMask , fill_mode='constant', cval=0., interpolation_order=0):
		# APPLIES TRANSLATION TO IMAGE SLIZE IN X AND Y DIRECTION

		"""Performs a random spatial shift of a Numpy image tensor.
		# Arguments
			x: Input tensor. Must be 3D.
			wrg: Width shift range, as a float fraction of the width.
			hrg: Height shift range, as a float fraction of the height.
			row_axis: Index of axis for rows in the input tensor.
			col_axis: Index of axis for columns in the input tensor.
			channel_axis: Index of axis for channels in the input tensor.
			fill_mode: Points outside the boundaries of the input
				are filled according to the given mode
				(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
			cval: Value used for points outside the boundaries
				of the input if `mode='constant'`.
			interpolation_order int: order of spline interpolation.
				see `ndimage.interpolation.affine_transform`
		# Returns
			Shifted Numpy image tensor.
		"""

				
		img = scipy.ndimage.shift(img, [tx, ty], mode=fill_mode, cval=cval, order=interpolation_order)

		return img




# class Generator2D(Sequence):

# 	def __init__(self, fileIDList, fileIDToPath, batchSize=32, dim=(224,224), nChannels=1, seed=1,
# 				 nClasses=10, shuffle=True, zoomRange=(1,1), rotationRange=0, widthShiftRange=0, heightShiftRange=0,
# 				 flipLR=False, flipUD=False):

# 		'Initialization'
# 		self.fileIDList = fileIDList
# 		self.fileIDToPath = fileIDToPath

# 		self.dim = dim
# 		self.batchSize = batchSize
# 		self.nChannels = nChannels
# 		self.nClasses = nClasses
# 		self.shuffle = shuffle
# 		self.random = np.random.RandomState(seed=seed)
# 		self.on_epoch_end()

# 		self.zoomRange = zoomRange
# 		self.rotationRange = rotationRange
# 		self.widthShiftRange = widthShiftRange
# 		self.heightShiftRange = heightShiftRange

# 		self.flipLR = flipLR
# 		self.flipUD = flipUD
		

# 	def __len__(self):
# 		'Denotes the number of batches per epoch'
# 		return int(np.floor(len(self.fileIDList) / self.batchSize))

# 	def __getitem__(self, index):
# 		'Generate one batch of data'
# 		# Generate indexes of the batch
# 		indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
# 		# Find list of IDs
# 		batchFileIDList = [self.fileIDList[k] for k in indexes]
# 		# Generate data
# 		X = self.__data_generation(batchFileIDList)
# 		#print(batchFileIDList)
# 		return X

# 	def on_epoch_end(self):
# 		self.indexes = np.arange(len(self.fileIDList))
# 		if self.shuffle:
# 			self.random.shuffle(self.indexes)

# 	def __data_generation(self, fileIDBatch):
# 		# TODO: MIGHT BE nChannels first
# 		'Generates data containing batchSize samples' # X : (n_samples, *dim, nChannels)
# 		# Initialization
		
# 		# Generate data
# 		for index, fileID in enumerate(fileIDBatch):
# 			img = self.getImageForFileID(fileID)
# 			#print("After returning: %0.2f" % np.max(img))
# 			if index == 0:
# 				X = np.empty((self.batchSize, self.nChannels, *self.dim), dtype=img.dtype)
# 			X[index,] = img
# 			#print("After in matrix: %0.2f" % np.max(X[index,]))
# 		#print(np.max(X))
# 		return X

# 	def getImageForFileID(self, fileID):

# 		sliceImg = imread(self.fileIDToPath[fileID])
# 		#print(sliceImg.dtype)
# 		#print("After reading: %0.2f" % np.max(sliceImg))
# 		# Get the random augmentation states
# 		#zf = self.getRandomZoomConfig(self.zoomRange)
# 		#theta = self.getRandomRotation(self.rotationRange)
# 		#tx, ty = self.getRandomShift(*self.dim, self.widthShiftRange, self.heightShiftRange)
# 		#sliceImg = self.applyZoom(sliceImg, zf)
# 		#sliceImg = self.applyRotation(sliceImg, theta)
# 		#sliceImg = self.applyShift(sliceImg, tx, ty)
		
# 		#if self.flipLR and self.getRandomFlipFlag():
# 		#	sliceImg = np.fliplr(sliceImg)
# 		#if self.flipUD and self.getRandomFlipFlag():
# 		#	sliceImg = np.fliPUD(sliceImg)

# 		#sliceImg = skimage.transform.resize(sliceImg, self.dim, order=1)
# 		sliceImg = np.array(PIL.Image.fromarray(sliceImg).resize(self.dim, resample=PIL.Image.BICUBIC))
# 		#print("After resize: %0.2f" % np.max(sliceImg))

# 		return sliceImg

# 	def getRandomFlipFlag(self):
# 		return self.random.choice([True, False])

# 	def getRandomZoomConfig(self, zoomRange):
# 		if zoomRange[0] == 1 and zoomRange[1] == 1:
# 			zf = 1
# 		else:
# 			zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
# 		return zf

# 	def applyZoom(self, img, zf, fill_mode='nearest', cval=0., interpolation_order=0):
# 		interp = cv2.INTER_CUBIC
# 		interpolation_order = 1
		
# 		origShape = img.shape
# 		img = scipy.ndimage.zoom(img, zf, mode=fill_mode, cval=cval, order=interpolation_order)
# 		if zf < 1:
# 			canvas = np.zeros(origShape, dtype=img.dtype)
# 			rowOffset = int(np.floor((origShape[0] - img.shape[0])/2))
# 			colOffset = int(np.floor((origShape[1] - img.shape[1])/2))
# 			canvas[rowOffset:(rowOffset+img.shape[0]), colOffset:(colOffset+img.shape[1])] = img
# 			img = canvas
# 		elif zf > 1:
# 			rowOffset = int(np.floor((img.shape[0] - origShape[0])/2))
# 			colOffset = int(np.floor((img.shape[1] - origShape[1])/2))
# 			img = img[rowOffset:(rowOffset+origShape[0]), colOffset:(colOffset+origShape[1])]

# 		img = np.array(PIL.Image.fromarray(img).resize(self.dim, resample=PIL.Image.BICUBIC))
# 		return img
	
# 	def getRandomRotation(self, rotationRange):
# 		theta = self.random.uniform(-rotationRange, rotationRange)
# 		return theta

# 	def applyRotation(self, img, theta, fill_mode='nearest', cval=0., interpolation_order=1):
# 		origShape = img.shape
# 		interp = cv2.INTER_NEAREST
# 		img = scipy.ndimage.rotate(img, theta, mode=fill_mode, cval=cval, order=interpolation_order, reshape=False)
# 		return img
# 	def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
# 		tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
# 		ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w
# 		return (tx, ty)

# 	def applyShift(self, img, tx, ty, fill_mode='nearest', cval=0., interpolation_order=0):
# 		img = scipy.ndimage.shift(img, [tx, ty], mode=fill_mode, cval=cval, order=interpolation_order)
# 		return img





# ###############################################################################
# ###############################################################################
# ###############     GENERATOR CLASSIFIER FUNCTIONS    #########################
# ###############################################################################
# ###############################################################################




# class Generator2DClassifier(Sequence):

# 	def __init__(self, fileIDList, fileIDToPath, fileIDToLabel, seed=1, batchSize=32, dim=(224,224), nChannels=1,
# 				 nClasses=10, shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, widthShiftRange=0, heightShiftRange=0,
# 				 flipLR = False, flipUD = False, customLength=-1, balanceClasses=True):

# 		'Initialization'
# 		self.fileIDList = fileIDList
# 		self.fileIDToPath = fileIDToPath

# 		self.dim = dim
# 		self.batchSize = batchSize
# 		self.nChannels = nChannels
# 		self.nClasses = nClasses
# 		self.shuffle = shuffle
# 		self.customLength = customLength
# 		self.balanceClasses = balanceClasses
# 		self.random = np.random.RandomState(seed=seed)
# 		self.on_epoch_end()

# 		self.zoomRange = zoomRange
# 		self.rotationRange = rotationRange
# 		self.widthShiftRange = widthShiftRange
# 		self.heightShiftRange = heightShiftRange

# 		self.flipLR = flipLR
# 		self.flipUD = flipUD
		
# 	def shuffleAndInitQueue(self):
# 		if self.shuffle:
# 			self.random.shuffle(self.fileIDList)
# 		# Create a list containing numClasses lists where list at
# 		# index n contains all fileIDs with label n
# 		idListByLabel = [[] for i in range(self.nClasses)]
# 		for fileID in self.fileIDList:
# 			label = self.fileIDToLabel[fileID]
# 			idListByLabel[label].append(fileID)
# 		# Repeat elements in shorter lists to make all lists the same size
# 		listLengths = [len(labelList) for labelList in idListByLabel]
# 		maxListLength = max(listLengths)
# 		for label in range(0, self.nClasses):
# 			if listLengths[label] == maxListLength:
# 				continue
# 			quotient, remainder = divmod(maxListLength, listLengths[label])
# 			idListByLabel[label] = (quotient * idListByLabel[label]) + (idListByLabel[label][:remainder])
# 		# Interleave elements in the lists now that all same size
# 		self.fileIDQueue = [val for tup in zip(*idListByLabel) for val in tup]
# 	def __len__(self):
# 		'Denotes the number of batches per epoch'
# 		#return int(np.floor(len(self.fileIDList) / self.batchSize))
# 		if self.customLength > 0:
# 			return self.customLength
# 		elif self.balanceClasses:
# 			return int(np.floor(len(self.fileIDQueue) / self.batchSize))
# 		else:
# 			return int(np.floor(len(self.fileIDList) / self.batchSize))

# 	def __getitem__(self, index):
# 		'Generate one batch of data'
# 		# Generate indexes of the batch
# 		if self.balanceClasses:
# 			indexes = list(range(index*self.batchSize, (index+1)*self.batchSize))
# 			batchFileIDList = [self.fileIDQueue[k] for k in indexes]
# 		else:
# 			indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
# 			batchFileIDList = [self.fileIDList[k] for k in indexes]

# 		# Generate data
# 		X = self.__data_generation(batchFileIDList)

# 		return X

# 	def on_epoch_end(self):
# 		if self.balanceClasses:
# 			if self.shuffle:
# 				self.shuffleAndInitQueue()
# 		else:
# 			self.indexes = np.arange(len(self.fileIDList))
# 			if self.shuffle:
# 				self.random.shuffle(self.indexes)


# 	def __data_generation(self, fileIDBatch):
# 		# TODO: MIGHT BE nChannels first
# 		'Generates data containing batchSize samples' # X : (n_samples, *dim, nChannels)
# 		# Initialization
# 		X = np.empty((self.batchSize, *self.dim, self.nChannels))
# 		y = np.empty((self.batchSize), dtype=int)
# 		# Generate data
# 		for index, fileID in enumerate(fileIDBatch):
# 			X[index, :, :, 0] = self.getImageForFileID(fileID)
# 			y[index] = self.fileIDToLabel[fileID]

# 		return X, keras.utils.to_categorical(y, num_classes=self.nClasses)

# 	def getImageForFileID(self, fileID):
# 		sliceImg = imread(self.fileIDToPath[fileID])
# 		# Get the random augmentation states
# 		zf = self.getRandomZoomConfig(self.zoomRange)
# 		theta = self.getRandomRotation(self.rotationRange)
# 		tx, ty = self.getRandomShift(*self.dim, self.widthShiftRange, self.heightShiftRange)
# 		sliceImg = self.applyZoom(sliceImg, zf)
# 		sliceImg = self.applyRotation(sliceImg, theta)
# 		sliceImg = self.applyShift(sliceImg, tx, ty)
		
# 		if self.flipLR and self.getRandomFlipFlag():
# 			sliceImg = np.fliplr(sliceImg)
# 		if self.flipUD and self.getRandomFlipFlag():
# 			sliceImg = np.fliPUD(sliceImg)
# 		sliceImg = np.array(PIL.Image.fromarray(sliceImg).resize(self.dim, resample=PIL.Image.BICUBIC))

# 		return sliceImg

# 	def getRandomFlipFlag(self):
# 		return self.random.choice([True, False])

# 	def getRandomZoomConfig(self, zoomRange):
# 		if zoomRange[0] == 1 and zoomRange[1] == 1:
# 			zf = 1
# 		else:
# 			zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
# 		return zf

# 	def applyZoom(self, img, zf, fill_mode='nearest', cval=0., interpolation_order=0):
# 		interp = cv2.INTER_CUBIC
# 		interpolation_order = 1
		
# 		origShape = img.shape
# 		img = scipy.ndimage.zoom(img, zf, mode=fill_mode, cval=cval, order=interpolation_order)
# 		if zf < 1:
# 			canvas = np.zeros(origShape, dtype=img.dtype)
# 			rowOffset = int(np.floor((origShape[0] - img.shape[0])/2))
# 			colOffset = int(np.floor((origShape[1] - img.shape[1])/2))
# 			canvas[rowOffset:(rowOffset+img.shape[0]), colOffset:(colOffset+img.shape[1])] = img
# 			img = canvas
# 		elif zf > 1:
# 			rowOffset = int(np.floor((img.shape[0] - origShape[0])/2))
# 			colOffset = int(np.floor((img.shape[1] - origShape[1])/2))
# 			img = img[rowOffset:(rowOffset+origShape[0]), colOffset:(colOffset+origShape[1])]

# 		img = np.array(PIL.Image.fromarray(img).resize(self.dim, resample=PIL.Image.BICUBIC))
# 		return img
	
# 	def getRandomRotation(self, rotationRange):
# 		theta = self.random.uniform(-rotationRange, rotationRange)
# 		return theta

# 	def applyRotation(self, img, theta, fill_mode='nearest', cval=0., interpolation_order=1):
# 		origShape = img.shape
# 		interp = cv2.INTER_NEAREST
# 		img = scipy.ndimage.rotate(img, theta, mode=fill_mode, cval=cval, order=interpolation_order, reshape=False)
# 		return img
# 	def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
# 		tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
# 		ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w
# 		return (tx, ty)

# 	def applyShift(self, img, tx, ty, fill_mode='nearest', cval=0., interpolation_order=0):
# 		img = scipy.ndimage.shift(img, [tx, ty], mode=fill_mode, cval=cval, order=interpolation_order)
# 		return img	






# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////

# ###############################################################################
# ###############################################################################
# ###############     GENERATOR CLASSIFIER FUNCTIONS    #########################
# ###############################################################################
# ###############################################################################

# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////




		
# class Generator3DClassifier(Sequence):
# 	# <<<<<<   THIS IS THE ONE NICK USED >>>>>>>>

# 	def __init__(self, fileIDList, fileIDToPath, 
# 				 fileIDToLabel, 
# 				 numFramesPerStack, seed=1, batchSize=32, dim=(200,200), nChannels=1,
# 				 nClasses=3, shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, 
# 				 widthShiftRange=0, heightShiftRange=0,
# 				 flipLR = False, flipUD = False, 

# 				 customLength=-1, balanceClasses=True):

# 	# def __init__(self, fileIDList, fileIDToPath, fileIDToLabel, numFramesPerStack, seed=seed, batchSize=batch_size, dim=dim, nChannels=nChannels,
# 	# 			 nClasses=num_classes, shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, widthShiftRange=0, heightShiftRange=0,
# 	# 			 flipLR = False, flipUD = False, customLength=-1, balanceClasses=True):


# 	# def __init__(self, fileIDList, fileIDToPath, fileIDToLabel, numFramesPerStack, seed=1, batchSize=32, dim=(224,224), nChannels=1,
# 	# 			 nClasses=10, shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, widthShiftRange=0, heightShiftRange=0,
# 	# 			 flipLR = False, flipUD = False, customLength=-1, balanceClasses=True):

# 		# balance class might be for classifier?

# 		'Initialization'
# 		self.fileIDList = fileIDList
# 		self.fileIDToPath = fileIDToPath
# 		self.fileIDToLabel = fileIDToLabel

# 		self.dim = dim
# 		self.batchSize = batchSize
# 		self.nChannels = nChannels
# 		self.nClasses = nClasses
# 		self.shuffle = shuffle
# 		self.random = np.random.RandomState(seed=seed)
# 		self.balanceClasses = balanceClasses
# 		self.on_epoch_end()
# 		self.numFramesPerStack = numFramesPerStack
# 		self.sepToken = sepToken

# 		self.zoomRange = zoomRange
# 		self.rotationRange = rotationRange
# 		self.widthShiftRange = widthShiftRange
# 		self.heightShiftRange = heightShiftRange
# 		self.customLength = customLength
		

# 		self.flipLR = flipLR
# 		self.flipUD = flipUD

# 	def shuffleAndInitQueue(self):
# 		if self.shuffle:
# 			self.random.shuffle(self.fileIDList)
# 		# Create a list containing numClasses lists where list at
# 		# index n contains all fileIDs with label n


# 		idListByLabel = [[] for i in range(self.nClasses)]
# 		for fileID in self.fileIDList:
# 			label = self.fileIDToLabel[fileID]
# 			idListByLabel[label].append(fileID)
# 		# Repeat elements in shorter lists to make all lists the same size
# 		listLengths = [len(labelList) for labelList in idListByLabel]
# 		#print("below")
# 		#print(listLengths)
# 		maxListLength = max(listLengths)
# 		for label in range(0, self.nClasses):
# 			if listLengths[label] == maxListLength:
# 				continue
# 			quotient, remainder = divmod(maxListLength, listLengths[label])
# 			idListByLabel[label] = (quotient * idListByLabel[label]) + (idListByLabel[label][:remainder])
# 		#for myList in idListByLabel:
# 		#	print(len(myList))
# 		# Interleave elements in the lists now that all same size
# 		self.fileIDQueue = [val for tup in zip(*idListByLabel) for val in tup]

# 	def __len__(self):
# 		'Denotes the number of batches per epoch'
# 		#return int(np.floor(len(self.fileIDList) / self.batchSize))
# 		if self.customLength > 0:
# 			return self.customLength
# 		elif self.balanceClasses:
# 			return int(np.floor(len(self.fileIDQueue) / self.batchSize))
# 		else:
# 			return int(np.floor(len(self.fileIDList) / self.batchSize))

# 	def __getitem__(self, index):
# 		'Generate one batch of data'
# 		# Generate indexes of the batch
# 		if self.balanceClasses:
# 			indexes = list(range(index*self.batchSize, (index+1)*self.batchSize))
# 			batchFileIDList = [self.fileIDQueue[k] for k in indexes]
# 		else:
# 			indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
# 			batchFileIDList = [self.fileIDList[k] for k in indexes]

# 		# Generate data
# 		X = self.__data_generation(batchFileIDList)

# 		return X

# 	def on_epoch_end(self):
# 		if self.balanceClasses:
# 			if self.shuffle:
# 				self.shuffleAndInitQueue()
# 		else:
# 			self.indexes = np.arange(len(self.fileIDList))
# 			if self.shuffle:
# 				self.random.shuffle(self.indexes)

# 	def __data_generation(self, fileIDBatch):
# 		# TODO: MIGHT BE nChannels first
# 		'Generates data containing batchSize samples' # X : (n_samples, *dim, nChannels)
# 		# Initialization
# 		X = np.empty((self.batchSize, *self.dim, self.numFramesPerStack, self.nChannels))
# 		y = np.empty((self.batchSize), dtype=int)
# 		# Generate data
# 		for index, fileID in enumerate(fileIDBatch):
# 			# Store sample
# 			#X[index,] = np.load(os.path.join(self.dirPath, fileName + '.npy'))
# 			X[index, :, :, :, 0] = self.getVolumeForFileID(fileID)

# 			y[index] = self.fileIDToLabel[fileID]
# 			#print(y[index])
# 			# Store class
# 		#print(y)
# 		#self.classes = np.concatenate((self.classes, y), axis=0)
# 		return X, keras.utils.to_categorical(y, num_classes=self.nClasses)








# 	def getVolumeForFileID(self, fileIDStart):
# 		#print(fileIDStart)
# 		"""
# 		Generates volume starting at given fileNameStart of height self.numFramesPerStack
# 		For example if fileNameStart is '00263523_054' (first part is MRN) and numFramesPerStack 
# 		is 5 then. The volume will be composed of images 00263523_054 through 00263523_058
# 		stacked on top of each other
# 		"""


# 		splitIndex = fileIDStart.rfind(self.sepToken)
# 		# This is equivalent to strfind in Matlab
# 		# finds index where the given string is. If it is not contained within it, returns -1

# 		if splitIndex == -1:
# 			raise ValueError("Separation token not found for fileID: " % fileIDStart)
# 			# So print an error if string fileIDStart does not contain sepToken

# 		fileIDBase = fileIDStart[0:(splitIndex)]
# 		# Split the string along the separation Token sepToken
# 		# So is string is aaaaaaaaaaabbbbb
# 		# fileIDBase becomes aaaaaaaaaaa and frameNumStartStr becomes bbbbbb

# 		frameNumStartStr = fileIDStart[(splitIndex+1):]


# 		numDigitsInNum = len(frameNumStartStr)

# 		#############
# 		#############
# 		#############     NOTE: THE FILES SHOULD BE NAMED LIKE
# 		############# 				PatientID_slicenum
# 		#############			with slicenum varying from 001 to the total number
# 		#############				of slices within the matrix
# 		#############           The sepToken is the string to split it at or "SEPARATION TOKEN"
# 		#############
# 		#############			SO FOR THIS ONE THE SEPTOKEN IS "_" before the slice numbers


# 		stack = np.empty( ( *self.dim, self.numFramesPerStack) , dtype=np.uint8 )

# 		# Initialize empty matrix of given shape and type without defining those values


# 		try:
# 			frameNumStart = int(frameNumStartStr)
# 		except ValueError:
# 			raise ValueError("Cannot cast %s to int for fileID %s" % (frameNumStartStr, fileIDStart))
		

# 		#########
# 		#########  Now apply random perturbations for data augmentation
# 		#########

# 		# Get the random augmentation states
# 		zf = self.getRandomZoomConfig(self.zoomRange)
# 		theta = self.getRandomRotation(self.rotationRange)
# 		tx, ty = self.getRandomShift(*self.dim, self.widthShiftRange, self.heightShiftRange)



# 		if self.flipLR:
# 			flipStackLR = self.getRandomFlipFlag()
# 		else:
# 			flipStackLR = False
# 		# IF random bool says so, flip left right

# 		if self.flipUD:
# 			flipStackUD = self.getRandomFlipFlag()
# 		else:
# 			flipStackUD = False
# 		# If random bool says so, flip matrix up down


# 		for index, frameNum in enumerate(range(frameNumStart, frameNumStart+self.numFramesPerStack)):

# 			fileNameNumStr = '{num:0{width}}'.format(num=frameNum, width=numDigitsInNum) 
# 			# Creates the string of the slice number that goes after sepToken
			
# 			fileID = "%s%s%s" % (fileIDBase, self.sepToken, fileNameNumStr)
# 			# Create string for corresponding file name

# 			sliceImg = np.asarray(Image.open(self.fileIDToPath[fileID]), dtype=np.uint8) # no dtype originally 
# 			# Open image with correct format
			
# 			sliceImg = cv2.resize(sliceImg, self.dim, interpolation=cv2.INTER_CUBIC)
# 			# Resize image to desired size

			
# 			# AUGMENTATION
# 			sliceImg = self.applyZoom(sliceImg, zf)
# 			sliceImg = self.applyRotation(sliceImg, theta) # Rotations only about slize Z direction. Makes sense why theres only one theta
# 			sliceImg = self.applyShift(sliceImg, tx, ty)

# 			if flipStackLR:
# 				sliceImg = np.fliplr(sliceImg)
# 			if flipStackUD:
# 				sliceImg = np.flipud(sliceImg)
			

# 			stack[:,:,index] = sliceImg
# 			#######
# 			#######   FINALLY, THROW THE CURRENT SLICE INTO THE RESULTANT 3D MATRIX "STACK"
# 			#######

# 		return stack




# 	def getRandomFlipFlag(self):
# 		return self.random.choice([True, False])
# 		# Defines random boolean, either True or False
# 		# Which are stored into functions for flipud and fliplr
# 		# so basically randomly orient the matrix in 3D


# 	def getRandomZoomConfig(self, zoomRange):
# 		if zoomRange[0] == 1 and zoomRange[1] == 1:
# 			zf = 1
# 		else:
# 			zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
# 		return zf




# 	def applyZoom(self, img, zf, fill_mode='nearest', cval=0., interpolation_order=0):
# 		interp = cv2.INTER_CUBIC
# 		interpolation_order = 1
# 		origShape = img.shape
# 		img = scipy.ndimage.zoom(img, zf, mode=fill_mode, cval=cval, order=interpolation_order)
# 		if zf < 1:
# 			canvas = np.zeros(origShape, dtype=img.dtype)
# 			rowOffset = int(np.floor((origShape[0] - img.shape[0])/2))
# 			colOffset = int(np.floor((origShape[1] - img.shape[1])/2))
# 			canvas[rowOffset:(rowOffset+img.shape[0]), colOffset:(colOffset+img.shape[1])] = img
# 			img = canvas
# 		elif zf > 1:
# 			rowOffset = int(np.floor((img.shape[0] - origShape[0])/2))
# 			colOffset = int(np.floor((img.shape[1] - origShape[1])/2))
# 			img = img[rowOffset:(rowOffset+origShape[0]), colOffset:(colOffset+origShape[1])]
# 		img = cv2.resize(img, origShape, interpolation=interp)
# 		return img
	


# 	def getRandomRotation(self, rotationRange):
# 		theta = self.random.uniform(-rotationRange, rotationRange)

# 		# Creates a random rotation vector which is later implemented
# 		# by the applyRotation function

# 		return theta

# 	def applyRotation(self, img, theta, fill_mode='nearest', cval=0., interpolation_order=1):
# 		"""Performs a random rotation of a Numpy image tensor.
# 		# Arguments
# 			x: Input tensor. Must be 3D.
# 			rg: Rotation range, in degrees.
# 			row_axis: Index of axis for rows in the input tensor.
# 			col_axis: Index of axis for columns in the input tensor.
# 			channel_axis: Index of axis for channels in the input tensor.
# 			fill_mode: Points outside the boundaries of the input
# 				are filled according to the given mode
# 				(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
# 			cval: Value used for points outside the boundaries
# 				of the input if `mode='constant'`.
# 			interpolation_order int: order of spline interpolation.
# 				see `ndimage.interpolation.affine_transform`
# 		# Returns
# 			Rotated Numpy image tensor.
# 		"""
		
# 		#x = self.apply_affine_transform(x, theta=theta, channel_axis=channel_axis,
# 		#						   fill_mode=fill_mode, cval=cval,
# 		#						   order=interpolation_order)
# 		origShape = img.shape
# 		interp = cv2.INTER_NEAREST
# 		img = scipy.ndimage.rotate(img, theta, mode=fill_mode, cval=cval, order=interpolation_order, reshape=False)
		
# 		# Applies the rotation to the input matrix

# 		return img


# 	def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
# 		tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
# 		ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w

# 		# Randomly defines the translation vector
# 		# which is the applied to matrix with the apply shift function

# 		return (tx, ty)


# 	def applyShift(self, img, tx, ty, fill_mode='nearest', cval=0., interpolation_order=0):
# 		"""Performs a random spatial shift of a Numpy image tensor.
# 		# Arguments
# 			x: Input tensor. Must be 3D.
# 			wrg: Width shift range, as a float fraction of the width.
# 			hrg: Height shift range, as a float fraction of the height.
# 			row_axis: Index of axis for rows in the input tensor.
# 			col_axis: Index of axis for columns in the input tensor.
# 			channel_axis: Index of axis for channels in the input tensor.
# 			fill_mode: Points outside the boundaries of the input
# 				are filled according to the given mode
# 				(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
# 			cval: Value used for points outside the boundaries
# 				of the input if `mode='constant'`.
# 			interpolation_order int: order of spline interpolation.
# 				see `ndimage.interpolation.affine_transform`
# 		# Returns
# 			Shifted Numpy image tensor.
# 		"""

# 		img = scipy.ndimage.shift(img, [tx, ty], mode=fill_mode, cval=cval, order=interpolation_order)
		
# 		# scipy ndimage shift
# 		#	shifts tensor using spline interpolation of requested order
# 		#	
# 		# 	inputs are:
# 		#		(input_matrix, shift, output=None, order = 3, mode = 'constant',
# 		#			cval=0.0 , prefiller = True )
# 		#				shift: float or seq. float is same for all axes, seq is one per
# 		#				
# 		#				output = array in which to place output
# 		#				order = order of spline interpolation
# 		#				fill_mode = determines boundaries. nearest or constant
# 		#				cval = scalar, value to fill past edges if mode is constant
# 		#				interpol

# 		return img




# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////
# ###### /////////////////////////////////////////////////////////////










if __name__ == "__main__":
		


	# use display TO CHECK THE DATA AUGMENTATION
	# NOTE: UP AND DOWN ARROW KEYS ALLOW FOR VIEWING
	study_name = '3D_SPGR_Segmentation'


	dirPath_train = "/d1/DeepLearning/3D_data/3D_data/train/"

	dirPath_valid = "/d1/DeepLearning/3D_data/3D_data/valid/"

	dirPath_train_mask = (dirPath_train + "mask/")
	dirPath_train_image = (dirPath_train + "image/")
	dirPath_valid_mask = (dirPath_valid + "mask/")
	dirPath_valid_image = (dirPath_valid + "image/")




	batch_size = 2 # 32
	epochs = 16 #10 

	total_images = 95
	total_train_images = 67
	total_valid_images = total_images - total_train_images

	steps_per_epoch = np.ceil( total_train_images / batch_size )
	validation_steps = np.ceil( total_valid_images / batch_size )

	num_train_to_generate = 2 * batch_size * steps_per_epoch
	num_valid_to_generate = 2 * batch_size *  validation_steps


	initial_epoch = 0 # SHOULD THIE BE 1?


	seed = 1
	numFramesPerStack = 60
	# NUM FRAMES PER STACK IS THE TOTAL NUMBER OF SLICES 


	input_shape = ( 1, 256 , 256 , numFramesPerStack ) # (1, 200, 200, 352)



	img_height = input_shape[ 1 ]
	img_width = input_shape[ 2 ]
	nChannels = input_shape[ 0 ]



	num_classes = 1 # NOTE: FOR SEGMENTATION, SEEMS UNNECESSARY. USED ONLY FOR CLASSIFIER


	# batch size*iterations = number of images




	total_n = batch_size * steps_per_epoch * epochs



	fileIDList_train_image, fileIDToPath_train_image, fileIDToLabel_train_image = getFileMappingForDir(dirPath_train_image, numFramesPerStack)
	fileIDList_length_train_image = len(fileIDList_train_image)


	train_image_generator = Generator3D(fileIDList_train_image, fileIDToPath_train_image, 
	    isMask = 0,
	    numFramesPerStack=numFramesPerStack, 
	    batchSize = batch_size , #batch_size , 
	    dim = ( img_height , img_width ) , nChannels = nChannels ,
	    seed = seed , shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, 
	    widthShiftRange=0, heightShiftRange=0, 
	    flipLR = False, flipUD = False )


	fileIDList_train_mask, fileIDToPath_train_mask, fileIDToLabel_train_mask = getFileMappingForDir(dirPath_train_mask, numFramesPerStack)
	fileIDList_length_train_mask = len(fleIDList_train_mask)


	train_mask_generator = Generator3D( fileIDList_train_mask, fileIDToPath_train_mask, 
	    isMask = 1,
	    numFramesPerStack=numFramesPerStack, 
	    batchSize = batch_size ,# batch_size , 
	    dim = ( img_height , img_width ) , nChannels = nChannels ,
	    seed = seed , shuffle=True, sepToken="_", zoomRange=(1,1), rotationRange=0, 
	    widthShiftRange=0, heightShiftRange=0, 
	    flipLR = False, flipUD = False )

	
	X = train_image_generator.__getitem__(0)
	print('dkjbnsvlkj;abdsbv;')
	print('SIZE IS ')
	print(np.shape(X))



