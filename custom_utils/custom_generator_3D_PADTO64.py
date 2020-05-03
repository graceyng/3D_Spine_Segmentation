




from keras.utils import Sequence
import numpy as np 
import glob
import os
import keras
import gc
import scipy
import cv2
import scipy.misc

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





class Generator_3D_PADTO64(Sequence):


	def __init__(self, fileIDList, fileIDToPath, numFramesPerStack,  batchSize=32, dim=(256,256), 
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
		
			# Img = np.empty((self.batchSize, self.nChannels, *self.dim, self.numFramesPerStack), dtype=int)
			# Mask = np.empty((self.batchSize, self.nChannels, *self.dim, self.numFramesPerStack), dtype=int)
	
			Img = np.empty((self.batchSize, self.nChannels, *self.dim, 64 ), dtype=int)
			Mask = np.empty((self.batchSize, self.nChannels, *self.dim, 64 ), dtype=int)
				
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
			

			Img_slice = cv2.resize( Img_slice , self.dim , interpolation=cv2.INTER_CUBIC ).astype( np.uint8 )
			Mask_slice = cv2.resize( Mask_slice , self.dim , interpolation=cv2.INTER_CUBIC ).astype( np.uint8 )
									
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

				Img_slice = np.flipud( Img_slice )
				Mask_slice = np.flipud( Mask_slice )

			if np.amax(Mask_slice)==255:
				Mask_slice = Mask_slice / 255


			Img_stack[ : , : , counter ] = Img_slice
			Mask_stack[ : , : , counter ] = Mask_slice
			

		# return  Img_stack , Mask_stack


		#######
		#######
		#######
		#######

		num_slices_to_pad = 64 - self.numFramesPerStack
		num_padded_slices_per_side = num_slices_to_pad / 2

		Img_stack_padded = np.zeros( ( *self.dim , 64 ) )
		Mask_stack_padded = np.zeros( ( *self.dim , 64 ) )

		Img_stack_padded[ : , : , 2:62 ] = Img_stack
		Img_stack_padded[ : , : , 0 ] = Img_stack[ : , : , 0 ] 
		Img_stack_padded[ : , : , 1 ] = Img_stack[ : , : , 0 ] 
		Img_stack_padded[ : , : , 62 ] = Img_stack[ : , : , 59 ]
		Img_stack_padded[ : , : , 63 ] = Img_stack[ : , : , 59 ]
		


		Mask_stack_padded[ : , : , 2:62 ] = Mask_stack
		Mask_stack_padded[ : , : , 0 ] = Mask_stack[ : , : , 0 ] 
		Mask_stack_padded[ : , : , 1 ] = Mask_stack[ : , : , 0 ] 
		Mask_stack_padded[ : , : , 62 ] = Mask_stack[ : , : , 59 ]
		Mask_stack_padded[ : , : , 63 ] = Mask_stack[ : , : , 59 ]

		img_mean = Img_stack.mean()
		img_std = Img_stack.std()

		rand_mat = np.random.normal( loc = img_mean , scale = img_std , size = Img_stack_padded.shape )
		rand_mat = rand_mat.clip( 0 , 255 ).astype( np.uint8 )

		Img_stack_padded = np.where( Img_stack_padded == 0 , rand_mat , Img_stack_padded )

		Img_stack_padded.astype(np.uint8)

		return Img_stack_padded , Mask_stack_padded



		#######
		#######
		#######
		#######



	def getRandomFlipFlag(self):


		return self.random.choice([True, False])



	def getRandomZoomConfig(self, zoomRange ):
		if zoomRange[0] == 1 and zoomRange[1] == 1:
			zf = 1
		else:
			zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
		return zf

	def applyZoom(self, img, zf, isMask , fill_mode='constant', cval=0., interpolation_order=0):
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

			## MUST BE NEAREST FOR MASK!!!!!!!
			img_out = scipy.ndimage.rotate( img , theta , reshape=False , order=3 , 
				mode='nearest' , cval=0 )

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



