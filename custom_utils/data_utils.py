import sys
sys.path.insert(0, "..\\")
import os
from custom_utils import file_utils
from scipy.misc import imread
from scipy.misc import imsave
from custom_utils import display
from custom_utils import print_utils
from custom_utils import file_utils
from custom_utils import dcm_utils
import time
import shutil
import nrrd
import glob
import math 
import cv2
import numpy as np
import nibabel as nib

def createOverlayDir(imgDirPathList, maskDirPathList, outputDirPath, outlineOnly=False, verbose=True, alpha=0.5, color=(0,255,0)):
	if os.path.isdir(outputDirPath):
		raise ValueError("Directory already exists: %s" % outputDirPath)
	os.mkdir(outputDirPath)

	if verbose:
		print("Collecting filepaths for each directory")
	imgFilePathList = []
	maskFilePathList = []
	for pathIndex in range(0, len(imgDirPathList)):
		imgDirPath = imgDirPathList[pathIndex]
		maskDirPath = maskDirPathList[pathIndex]

		imgFilePathListSubset = file_utils.getFilePathsFromDir(imgDirPath)
		maskFilePathListSubset = file_utils.getFilePathsFromDir(maskDirPath)
		imgFileNameListSubset = [os.path.basename(os.path.normpath(path)) for path in imgFilePathListSubset]
		maskFileNameListSubset = [os.path.basename(os.path.normpath(path)) for path in maskFilePathListSubset]
		if len(set(imgFileNameListSubset)) != len(set(maskFileNameListSubset)):
			raise ValueError("imgFileNameList and maskFileNameList not equal for %s %s" % (imgDirPath, maskDirPath))
		imgFilePathList += imgFilePathListSubset
		maskFilePathList += maskFilePathListSubset
	if verbose:
		print("Finished collecting filepaths")
		startTime = time.time()

	for pathIndex, path in enumerate(imgFilePathList):
		if verbose:
			if pathIndex > 0:
				if pathIndex % 200 == 0:
					ellapsedTime = time.time() - startTime
					timePerFile = ellapsedTime/pathIndex
					timeRemaining = timePerFile * (len(imgFilePathList) - pathIndex)
					timeRemainingString = print_utils.getTimeString(timeRemaining)
					print("Processing %d of %d, %s time remaining" % (pathIndex+1, len(imgFilePathList), timeRemainingString))
			else:
				print("Processing %d of %d" % (pathIndex+1, len(imgFilePathList)))
		imgFilePath = imgFilePathList[pathIndex]
		maskFilePath = maskFilePathList[pathIndex]
		img = imread(imgFilePath)
		mask = imread(maskFilePath)
		if outlineOnly:
			overlayImg = display.getGrayWithOutline(img, mask, alpha=alpha, color=color)
		else:
			overlayImg = display.getGrayWithOverlay(img, mask, alpha=alpha, color=color)
		fileName = os.path.basename(os.path.normpath(imgFilePath))
		outputFilePath = os.path.join(outputDirPath, fileName)
		if os.path.exists(outputFilePath):
			raise ValueError("Duplicate file: %s" % outputFilePath)
		imsave(outputFilePath, overlayImg)

	
def getNumberInEachFolder(dirPath):
	dirNameList = file_utils.getDirNamesFromDir(dirPath)
	numFilesForDirList = []
	for dirName in dirNameList:
		viewDirPath = os.path.join(dirPath, dirName)
		numFilesForDirList.append(len([name for name in os.listdir(viewDirPath + "\\.")]))
	result = zip(dirNameList, numFilesForDirList)
	result = sorted(result, key=lambda x: x[1], reverse=True)
	return result

# Delete a particular class from all sets if less than certain number files in training set
def remveSubDirsWithLessThanNumFiles(dirPathRoot, numFilesRequired, setNameList = ["train","valid","test"]):
	result = getNumberInEachFolder(os.path.join(dirPathRoot, setNameList[0]))

	for dirName, numFiles in result:
		if numFiles < numFilesRequired:
			print("Deleting %s with %d files" % (dirName, numFiles))
			for setName in setNameList:
				pathToDelete = os.path.join(dirPathRoot, setName, dirName)
				shutil.rmtree(pathToDelete)

def verifySlicerData(rootDirPath, segDirName="seg", printErrors=True, labelQuery="*label.nrrd"):
	patientDirNameList = file_utils.getDirNamesFromDir(rootDirPath)

	goodPatientDirPathList = []
	goodSeriesDirPathList = []
	goodLabelMapFilePathList = []
	badDirPathList = []

	totalErrorMsg = ""
	for patientIndex, patientDirName in enumerate(patientDirNameList):
		patientDirPath = os.path.join(rootDirPath, patientDirName)
		seriesDirNameList = file_utils.getDirNamesFromDir(patientDirPath)
		errorMsg = ""
		numSlices = -1
		numSliceImg = -1
		# If no directories then this could be NIFTI format
		if len(seriesDirNameList) == 0:
			fileNameList = file_utils.getFileNamesFromDir(patientDirPath, ext=".nii.gz")
			fileNameList = [fileName for fileName in fileNameList if "mask" not in fileName.lower()]
			if len(fileNameList) == 0:
				errorMsg += "No image NIFTI found for: %s\n" % (patientDirName)
			elif len(fileNameList) > 1:
				errorMsg += "Multiple NIFTI files found for: %s\n" % (patientDirName)
			else:
				# Misnomer but this is the nifti file path
				seriesDirPath = os.path.join(patientDirPath, fileNameList[0])
				img = nib.load(seriesDirPath)
				numSliceImg = img.header["dim"][3]
			
			labelFilePathList = glob.glob(os.path.join(patientDirPath, labelQuery))
			print(os.path.join(patientDirPath, labelQuery))
			if len(labelFilePathList) > 1:
				errorMsg += "Multiple labelmap segmentations detected, only one expected for %s\n" % patientDirName
			elif len(labelFilePathList) == 0:
				errorMsg += "No labelmap detected for %s\n" % patientDirName
			else:
				labelFilePath = labelFilePathList[0]
				header = nrrd.read_header(labelFilePath)
				numSlices = header["sizes"][2]
				if numSliceImg != -1 and numSlices != numSliceImg:
					errorMsg += "%d slices in image but %d in labelmap for %s\n" % (numSliceImg, numSlices, patientDirName)


		else:
			# Check for seg folder
			if "seg" not in seriesDirNameList:
				errorMsg += "No '%s' folder for: %s\n" % (segDirName, patientDirName)
			# If present get number of slices in labelmap
			else:
				segDirPath = os.path.join(patientDirPath, segDirName)
				labelFilePathList = glob.glob(os.path.join(segDirPath, labelQuery))
				if len(labelFilePathList) > 1:
					errorMsg += "Multiple labelmap segmentations detected, only one expected for %s\n" % patientDirName
				elif len(labelFilePathList) == 0:
					errorMsg += "No labelmap detected for %s\n" % patientDirName
				else:
					labelFilePath = labelFilePathList[0]
					header = nrrd.read_header(labelFilePath)
					numSlices = header["sizes"][2]
			if len(seriesDirNameList) != 2:
				errorMsg += "Incorrect number of series folders for %s, expected 2 received %d\n" % (patientDirName, len(seriesDirNameList))
				for seriesDirName in seriesDirNameList:
					if seriesDirName == segDirName:
						continue
					seriesDirPath = os.path.join(patientDirPath, seriesDirName)
					numFiles = len(file_utils.getFileNamesFromDir(seriesDirPath))
					# If we were able to get number of slices from labelmap, check if each series has correct number
					if numSlices != -1:
						if numSlices == numFiles:
							errorMsg += "%s has correct number of files for\n" % seriesDirName
						else:
							errorMsg += "%s has incorrect number of files for\n" % seriesDirName
			elif numSlices != -1:
				for seriesDirName in seriesDirNameList:
					if seriesDirName == segDirName:
						continue
					seriesDirPath = os.path.join(patientDirPath, seriesDirName)
					numFiles = len(file_utils.getFileNamesFromDir(seriesDirPath))
					if numSlices != numFiles:
						errorMsg += "%s has incorrect number of files for %s\n" % (seriesDirName, patientDirName)
		if errorMsg != "":
			errorMsg += "----------\n"
			totalErrorMsg += errorMsg
			badDirPathList.append(patientDirPath)
		else:
			goodPatientDirPathList.append(patientDirPath)
			goodSeriesDirPathList.append(seriesDirPath)
			goodLabelMapFilePathList.append(labelFilePath)
	if printErrors:		
		print(totalErrorMsg)
	return totalErrorMsg, badDirPathList, goodPatientDirPathList, goodSeriesDirPathList, goodLabelMapFilePathList

def convertImages(dcmFolderPath, patientID, outputDirPath, outputDims, width, level, outputNameToken="axial"):
	# Check if nifti file
	if os.path.isfile(dcmFolderPath.encode("UTF-8")):
		stack = nib.load(dcmFolderPath).get_fdata()
		numSlices = stack.shape[2]
		for zIndex in range(0, stack.shape[2]):
			img = stack[:,:,zIndex]
			img = dcm_utils.thresholdImage(img, level, width, rescaleSlope=1, rescaleIntercept=0)
			img = cv2.resize(img, outputDims, interpolation=cv2.INTER_CUBIC)

			outputFileName = "%s_%s_%04d.png" % (outputNameToken, patientID, zIndex)
			outputFilePath = os.path.join(outputDirPath, outputFileName)
			cv2.imwrite(outputFilePath, img)
		
	else:
		dcmFilePathList = file_utils.getFilePathsFromDir(dcmFolderPath)
		dcmFilePathList = dcm_utils.sortDCMFilePaths(dcmFilePathList)
		numSlices = len(dcmFilePathList)
		for pathIndex, filePath in enumerate(dcmFilePathList):
			(img, rescaleSlope, rescaleIntercept) = dcm_utils.loadDICOMAsNumpy(filePath)
			img = dcm_utils.thresholdImage(img, level, width, rescaleSlope, rescaleIntercept)
			img = cv2.resize(img, outputDims, interpolation=cv2.INTER_CUBIC)

			outputFileName = "%s_%s_%04d.png" % (outputNameToken, patientID, pathIndex)
			outputFilePath = os.path.join(outputDirPath, outputFileName)
			cv2.imwrite(outputFilePath, img)
		
	return numSlices
def convertSegmentations(nrrdFilePath, patientID, outputDirPath, outputDims, outputNameToken="axial"):
	if not os.path.isfile(nrrdFilePath):
		raise ValueError("No file found at: %s" % nrrdFilePath)

	data, header = nrrd.read(nrrdFilePath)
	if header['space'] != 'left-posterior-superior':
		raise ValueError("Unexpected orientation")

	# By default in 0 and 1's. Convert to 0's and 255's
	data = data * 255.0/data.max()
	if header["space directions"][0][0] != 0:
		data = np.rot90(data, axes=(1,0))
		data = np.fliplr(data)
	data = cv2.resize(data, outputDims, interpolation=cv2.INTER_NEAREST)
	numSlices = data.shape[2]
	for index in range(0, numSlices):
		outputFileName = "%s_%s_%04d.png" % (outputNameToken, patientID, index)
		outputFilePath = os.path.join(outputDirPath, outputFileName)
		cv2.imwrite(outputFilePath, data[:,:,index])
	return numSlices

def generateCTSegmentationTrainingData(outputDirPath, dataDirPath, outputDims, width, level,
										trainDirName="train", validDirName="valid", 
										testDirName="test", imgDirName="image", maskDirName="mask", 
										classDirName="dummy_class", hasTestSet=False, breakdown=[0.8, 0.2],
										segDirName="seg", outputNameToken="axial", labelQuery="*label.nrrd"):
	outputRootPath, outputDirName = os.path.split(os.path.normpath(outputDirPath))
	outputDirPath, outputDirName = file_utils.getUniqueDirPath(outputRootPath, outputDirName)

	print("Output will be written to: %s" % outputDirPath)

	os.makedirs(os.path.join(outputDirPath, trainDirName, imgDirName, classDirName))
	os.makedirs(os.path.join(outputDirPath, trainDirName, maskDirName, classDirName))
	os.makedirs(os.path.join(outputDirPath, validDirName, imgDirName, classDirName))
	os.makedirs(os.path.join(outputDirPath, validDirName, maskDirName, classDirName))

	if hasTestSet:
		os.makedirs(os.path.join(outputDirPath, testDirName, imgDirName, classDirName))
		os.makedirs(os.path.join(outputDirPath, testDirName, maskDirName, classDirName))


	if hasTestSet:
		if len(breakdown) != 3:
			raise ValueError("Expected 'breakdown' length of 3 but received %d" % len(breakdown))
	else:
		if len(breakdown) != 2:
			raise ValueError("Expected 'breakdown' length of 2 but received %d" % len(breakdown))
	if sum(breakdown) != 1:
		raise ValueError("Sum of breakdown does not equal '1', but instead equals %0.2f" % sum(breakdown))

	# Validate data
	print("==============================")
	print("Checking source data for errors")
	
	(errorMsg, badDirPathList, goodPatientDirPathList, 
	goodSeriesDirPathList, goodLabelMapFilePathList) =  verifySlicerData(dataDirPath, printErrors=False, segDirName=segDirName, labelQuery=labelQuery)

	errorMsg = "----------\n" + errorMsg 
	if errorMsg == "":
		errorMsg = "NONE FOUND\n----------"
	print("Found following errors:\n\n%s" % errorMsg)
	print("%d correctly formatted patients, %d incorrect" % (len(goodPatientDirPathList), len(badDirPathList)))
	if len(goodPatientDirPathList) == 0:
		print("Process terminated as no paths to process")
		return
	print("==============================")

	# Determine what set output will be in - train/valid/test(optional)
	if hasTestSet:
		numTrain = int(np.ceil(len(goodPatientDirPathList) * breakdown[0]))
		numValid = int(np.round(len(goodPatientDirPathList) * breakdown[1]))
		numTest = len(goodPatientDirPathList) - numTrain - numValid
		setNameList = [trainDirName] * numTrain + [validDirName] * numValid + [testDirName] * numTest
		print("Study Breakdown: %d %s, %d %s, %d %s" % (numTrain, trainDirName, numValid, validDirName, numTest, testDirName))
	else:
		numTrain = math.ceil(len(goodPatientDirPathList) * breakdown[0])
		numValid = len(goodPatientDirPathList) - numTrain
		setNameList = [trainDirName] * numTrain + [validDirName] * numValid
	print("Generating training data")
	# Output text files indicating which patients in each
	for patientIndex, patientDirPath in enumerate(goodPatientDirPathList):
		outputFilePath = os.path.join(outputDirPath, "%s_dirs.txt" % setNameList[patientIndex])
		with open(outputFilePath, 'a') as outputFile:
			outputFile.write("%s\n" % patientDirPath)
	# Generate training data
	startTime = time.time()
	for patientIndex, patientDirPath in enumerate(goodPatientDirPathList):
		patientDirName = os.path.basename(os.path.normpath(patientDirPath))
		# Print update on computation time
		if patientIndex == 0:
			print("Processing 1 of %d" % len(goodPatientDirPathList))
		else:
			ellapsedTime = time.time() - startTime
			timePerPatient = ellapsedTime / patientIndex
			remainingTime = timePerPatient * (len(goodPatientDirPathList) - patientIndex)
			print("%s ellapsed, %s remaining - %d of %d" % (print_utils.getTimeString(ellapsedTime),
													print_utils.getTimeString(remainingTime),
													patientIndex+1, len(goodPatientDirPathList)))
		seriesDirPath = goodSeriesDirPathList[patientIndex]
		labelMapFilePath = goodLabelMapFilePathList[patientIndex]

		imgOutputDirPath = os.path.join(outputDirPath, setNameList[patientIndex], imgDirName, classDirName)
		maskOutputDirPath = os.path.join(outputDirPath, setNameList[patientIndex], maskDirName, classDirName)

		numImages = convertImages(seriesDirPath, patientDirName, imgOutputDirPath, outputDims, width, level, outputNameToken=outputNameToken)
		numMasks = convertSegmentations(labelMapFilePath, patientDirName, maskOutputDirPath, outputDims, outputNameToken=outputNameToken)
		if numImages != numMasks:
			print("Inconsistent for %s - %d and %d" % (patientID, numImages, numMasks))


if __name__ == "__main__":
	pass
	#rootDirPath = "J:\\Chirinos Projects 6 Penn Tower Team\\Team Member Folders\\Bilal\\Aortic segmentation non-contrast CT"
	#verifySlicerData(rootDirPath)

	# Aorta
	#outputDirPath = "D:\\data\\temp_test"
	#dataDirPath = "J:\\Chirinos Projects 6 Penn Tower Team\\Team Member Folders\\Bilal\\Aortic segmentation non-contrast CT"
	#generateCTSegmentationTrainingData(outputDirPath, dataDirPath, outputDims=(256,256), width=150, level=30)
	
	# Spleen
	#outputDirPath = "D:\\data\\spleen_output"
	#dataDirPath = "J:\\Chirinos Projects 6 Penn Tower Team\\PMBB Segmentations\\Liver Scans"
	#generateCTSegmentationTrainingData(outputDirPath, dataDirPath, outputDims=(256,256), width=150, level=30, labelQuery="*spleen-label*.nrrd")