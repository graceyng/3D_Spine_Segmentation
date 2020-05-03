import glob
import os
import nibabel as nib 
import numpy as np
import pydicom
import cv2
import dicom2nifti

def sortDCMFilePaths(dcmFilePaths, direction=0):
	sliceLocations = []
	for index in range(0, len(dcmFilePaths)):
		filePath = dcmFilePaths[index]
		dcmImg = pydicom.dcmread(filePath, specific_tags=["ImagePositionPatient"])
		zCoord = dcmImg.ImagePositionPatient[2]
		sliceLocations.append(zCoord)
	# Z-Coordinate increases as we go towards head of patient so sort increasing
	# so that our stack starts inferiorly and moves superiorly
	sortedInds = np.argsort(sliceLocations)
	# Use mapping from above to sort the filenames
	dcmFilePaths = np.array(dcmFilePaths)
	dcmFilePaths = dcmFilePaths[sortedInds]
	return dcmFilePaths
def resizeStack(stack, targetSize, method, zBatch=150):
	output = np.empty((*targetSize, stack.shape[2]))
	for zIndex in range(0, stack.shape[2], zBatch):
		maxZ = (min(zIndex+zBatch, stack.shape[2]))
		sample = stack[:,:,zIndex:maxZ]
		sample = cv2.resize(sample, (targetSize[1], targetSize[0]), interpolation=method)
		if len(sample.shape) == 2:
			sample = sample[:,:,np.newaxis]
		output[:,:,zIndex:maxZ] = sample
	return output
def loadDICOMAsNumpy(dcmFilePath):
	dcmImg = pydicom.dcmread(dcmFilePath)
	rescaleSlope = dcmImg.RescaleSlope
	rescaleIntercept = dcmImg.RescaleIntercept
	dims = (int(dcmImg.Rows), int(dcmImg.Columns))
	npImg = np.zeros(dims, dtype=dcmImg.pixel_array.dtype)
	npImg[:,:] = dcmImg.pixel_array
	return (npImg, rescaleSlope, rescaleIntercept)

def thresholdImage(img, level, width, rescaleSlope=1, rescaleIntercept=0):
	img = img.astype(np.float, copy=False)
	img = img * rescaleSlope + rescaleIntercept

	img =  np.piecewise(img, 
		[img <= (level - 0.5 - (width-1)/2),
		img > (level - 0.5 + (width-1)/2)],
		[0, 255, lambda img: ((img - (level - 0.5))/(width-1) + 0.5)*(255-0)])
	img = img.astype(np.uint8, copy=False)
	return img

def getRawStackFromDCMDirPath(dcmDirPath, query="*.dcm", direction=0):
	dcmFilePathList = glob.glob(os.path.join(dcmDirPath, query))
	dcmFilePathList = sortDCMFilePaths(dcmFilePathList, direction=direction)
	if len(dcmFilePathList) == 0:
		raise ValueError("No files found in that directory: %s" % dcmDirPath)
	for dcmFileIndex, dcmFilePath in enumerate(dcmFilePathList):
		npImg, rescaleSlope, rescaleIntercept = loadDICOMAsNumpy(dcmFilePath)
		if dcmFileIndex == 0:
			stack = np.empty((npImg.shape + (len(dcmFilePathList),)))
		npImg = npImg * rescaleSlope + rescaleIntercept
		stack[:, :, dcmFileIndex] = npImg		
	return stack


def getStackFromDCMDirPath(dcmDirPath, sliceSize, level, width, direction=0, query="*.dcm"):
	dcmFilePathList = glob.glob(os.path.join(dcmDirPath, query))
	dcmFilePathList = sortDCMFilePaths(dcmFilePathList, direction=direction)
	
	stack = np.empty(sliceSize + (len(dcmFilePathList),), dtype=np.uint8)

	for dcmFileIndex, dcmFilePath in enumerate(dcmFilePathList):
		npImg, rescaleSlope, rescaleIntercept = loadDICOMAsNumpy(dcmFilePath)
		sliceImg = thresholdImage(npImg, level, width, rescaleSlope, rescaleIntercept)
		sliceImg = cv2.resize(sliceImg, sliceSize, interpolation=cv2.INTER_CUBIC)
		stack[:, :, dcmFileIndex] = sliceImg
	return stack


def dcmToNII_CT(dcmDirPath, niiOutputPath, sliceSize, level, width):
	def processMethod(img, rescaleSlope, rescaleIntercept):
		return thresholdImage(img, level, width, rescaleSlope, rescaleIntercept)

	dcmToNII(dcmDirPath, niiOutputPath, sliceSize, processMethod)
def getNIIAffine(pixDim):

	pixDimMat = np.array([[pixDim[0],0,0,0],[0,pixDim[1],0,0],[0,0,pixDim[2],0],[0,0,0,1]])
	rotate = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
	reflect = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
	affine = np.matmul(pixDimMat, np.matmul(rotate, reflect))
	return affine
# see doc here: https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
# pixelSpacing = (x_width, y_width, z_width)
'''
def getNIIHeader(scanDim, pixelSpacing):
	header = nib.Nifti1Header()
	header["dim"] = np.array([len(scanDim)] + scanDim + np.repeat(1, 8 - len(scanDim)).tolist())
	header["pixdim"] = np.array([len(pixelSpacing)] + pixelSpacing + np.repeat(1, 8 - len(pixelSpacing)).tolist())
'''

def dcmFilePathListToNII(dcmFilePathList):
	#print(dcmFilePathList)
	dcmFilePathList = sortDCMFilePaths(dcmFilePathList)
	dcmImg = pydicom.dcmread(dcmFilePathList[0], specific_tags=["PatientPosition"])
	#print(dcmImg)
	'''
	headFirst = False
	noTag = False
	if hasattr(dcmImg, "PatientPosition"):
		if dcmImg.PatientPosition.lower() == "hfs":
			headFirst = True
	else:
		if hasattr(dcmImg, "PatientGantryRelationshipCodeSequence"):
			gantry = dcmImg.PatientGantryRelationshipCodeSequence
			if len(gantry) > 0:
				if hasattr(gantry[0], "CodeMeaning"):
					if gantry[0].CodeMeaning.lower() == "headfirst":
						headFirst = True
					else:
						headFirst = False
				else:
					#print("NONE FOUND - 0")
					noTag = True
			else:
				#print("NONE FOUND - 1")
				noTag = True
		else:
			#print("NONE FOUND - 2")
			noTag = True
	if headFirst:
		pass
		#print("Head first")
	else:
		pass
		#print("Feet first")
	'''
	dcmFirst = pydicom.dcmread(dcmFilePathList[0], specific_tags=["SliceThickness", "PixelSpacing", "ImagePositionPatient"])
	dcmSecond = pydicom.dcmread(dcmFilePathList[1], specific_tags=["SliceThickness", "PixelSpacing", "ImagePositionPatient"])
	#print(dcmFirst.ImagePositionPatient)
	#print(dcmSecond.ImagePositionPatient)
	sliceThickness = np.abs(dcmSecond.ImagePositionPatient[2] - dcmFirst.ImagePositionPatient[2])
	pixDim = [float(dcmFirst.PixelSpacing[0]), float(dcmFirst.PixelSpacing[1]), sliceThickness]
	#print(pixDim)
	affine = getNIIAffine(pixDim)

	rescaleSlope = None
	rescaleIntercept = None
	for dcmFileIndex, dcmFilePath in enumerate(dcmFilePathList):
		#print(dcmFilePath)
		npImg, sliceSlope, sliceIntercept = loadDICOMAsNumpy(dcmFilePath)
		if dcmFileIndex > 0 and (rescaleSlope != sliceSlope or rescaleIntercept != sliceIntercept):
			pass
			#print("Rescaleslope or intercept not consistent for dcm dir path: %s" % dcmDirPath)
		rescaleSlope = sliceSlope
		rescaleIntercept = sliceIntercept
		npImg = npImg * rescaleSlope + rescaleIntercept
		if dcmFileIndex == 0:
			stack = np.empty(npImg.shape + (len(dcmFilePathList),), dtype=npImg.dtype)
		stack[:, :, dcmFileIndex] = npImg
	#print(stack.shape)
	#header = nib.Nifti1Header()
	#header.set_slope_inter(rescaleSlope, rescaleIntercept)
	niiStack = nib.Nifti1Image(stack, affine)
	#niiStack.to_filename(niiOutputPath)
	return niiStack

def dcmToNIISimple(dcmDirPath, niiOutputPath, queryToken="*.dcm"):
	dcmFilePathList = glob.glob(os.path.join(dcmDirPath, queryToken))
	#print(dcmFilePathList)
	stack =  dcmFilePathListToNII(dcmFilePathList)
	stack.to_filename(niiOutputPath)



# NIB header manip from https://github.com/nipy/nibabel/blob/master/nibabel/nifti1.py (search "set_slope")
def dcmToNII(dcmDirPath, niiOutputPath, sliceSize, processMethod):
	dcmFilePathList = glob.glob(os.path.join(dcmDirPath, "*.dcm"))
	dcmFilePathList = sortDCMFilePaths(dcmFilePathList)
	
	stack = np.empty(sliceSize + (len(dcmFilePathList)), dtype=np.uint8)


	for dcmFileIndex, dcmFilePath in enumerate(dcmFilePathList):
		npImg, rescaleSlope, rescaleIntercept = loadDICOMAsNumpy(dcmFilePath)

		sliceImg = processMethod(npImg, rescaleSlope, rescaleIntercept)
		sliceImg = cv2.resize(sliceImg, sliceSize, interpolation=cv2.INTER_CUBIC)
		stack[:, :, dcmFileIndex] = sliceImg


	niiStack = nib.Nifti1Image(stack, np.eye(4))
	print("Writing output")
	niiStack.to_filename(niiOutputPath)

if __name__ == "__main__":
	dcmDirPath = "J:\\Chirinos Projects 6 Penn Tower Team\\Liver Scans\\16317654\\CT 5MM AP"	
	dicom2nifti.dicom_series_to_nifti(dcmDirPath, "test.nii.gz", reorient_nifti=True)
	#dcmFilePathList = glob.glob(os.path.join(dcmDirPath, "*.dcm"))
	#dcmFilePathList = sortDCMFilePaths(dcmFilePathList)

	#dcmFirst = pydicom.dcmread(dcmFilePathList[0], specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"])
	#dcmLast = pydicom.dcmread(dcmFilePathList[-1], specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"])
	#affine = dicom2nifti.common.create_affine([dcmFirst, dcmLast])

	#dcmToNIIRaw(dcmDirPath, niiOutputPath="output.nii.gz")
