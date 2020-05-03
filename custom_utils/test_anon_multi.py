import dcmstack
import glob
import os
import gdcm
import pydicom
import dicom2nifti
import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileDataset
import csv_utils
import random 
import csv
import traceback 
import json 
import time 
from print_utils import getTimeString

def getOrientationForFileList(dcmFileList):
	dcmFile = dcmFileList[0]
	if not hasattr(dcmFile, "ImageOrientationPatient"):
		return None
	values = dcmFile.ImageOrientationPatient
	#values = token.split("\\")
	if len(values) != 6:
		return None
	values = [round(float(item)) for item in values]
	plane = np.cross(values[0:3], values[3:6])
	plane = [abs(item) for item in plane]
	if plane[0] == 1:
		return "SAGITTAL"
	elif plane[1] == 1:
		return "CORONAL"
	elif plane[2] == 1:
		return "AXIAL"
	return None
def getSliceThicknessForSortedFiles(dcmFileList, orientationToken):
	orientationMapping = {
		"SAGITTAL": 0,
		"CORONAL": 1,
		"AXIAL": 2
	}
	if orientationToken in orientationMapping:
		orientation = orientationMapping[orientationToken]
	else:
		return None
	if len(dcmList) <= 1:
		return None
	dcmFirst = dcmList[0]
	dcmSecond = dcmList[1]
	if not hasattr(dcmFirst, "ImagePositionPatient") or not hasattr(dcmSecond, "ImagePositionPatient"):
		return None
	sliceThickness = abs(float(dcmSecond.ImagePositionPatient[orientation]) - float(dcmFirst.ImagePositionPatient[orientation]))
	return sliceThickness

tagExcludeDict = {
	"FloatPixelData": (0x7fe0, 0x10)
}
tagExcludeList = [tagExcludeDict[key] for key in tagExcludeDict]
def dictify(ds):
	"""Turn a pydicom Dataset into a dict with keys derived from the Element tags.

	Parameters
	----------
	ds : pydicom.dataset.Dataset
		The Dataset to dictify

	Returns
	-------
	output : dict
	"""
	output = dict()
	for elem in ds:
		if elem.tag in tagExcludeList:
			continue
		if elem.VR != 'SQ':
			if type(elem.value) == pydicom.multival.MultiValue:
				output[elem.tag] = list(elem.value)
				output[elem.tag] = [str(item) for item in output[elem.tag]]
			else:
				output[elem.tag] = str(elem.value)
		else:
			output[elem.tag] = [dictify(item) for item in elem]
	return output

dicom2nifti.settings.disable_validate_slicecount()
dicom2nifti.settings.disable_validate_orientation()
dicom2nifti.settings.disable_validate_orthogonal()


dcmDirPath = "D:\\data\\0ground_truth_data\\abdominal_wall\\10512224\\CT DE_Abd_Pel 5.0 Br40 3 F_0.6"
outputFilePath = "D:\\test_transfer\\output.nii.gz"
outputDirPath = "D:\\test_transfer\\"
dcmDirPath = "D:\\test_transfer\\1.2.124.113532.170.212.54.51.20021206.100934.4357501"

outputDirPath = "D:\\test_transfer\\AnonymizedTest\\data"


seriesUIDTag = gdcm.Tag(0x0020,0x000E)
imageTypeTag = gdcm.Tag(0x0008, 0x0008)




rootDirPath = "D:\\test_transfer\\LiverDataTest"

empiMapFilePath = "D:\\data\\pmbb_info\\master_pmbb_list_uniqueids_full_gt.csv"
empiToPacketUUID = csv_utils.getDictionaryFromCSV(empiMapFilePath, "EMPI","PACKET_UUID")

accessionMapFilePath = "D:\\data\\pmbb_info\\accession_to_pmbb.csv"
accessionToPMBB = csv_utils.getDictionaryFromCSV(accessionMapFilePath,"PENN_ACCESSION","PMBB_ACCESSION")

seriesUIDMapFilePath = "D:\\data\\pmbb_info\\seriesuid_to_pmbbseriesuid.csv"
seriesUIDToPMBB = csv_utils.getDictionaryFromCSV(seriesUIDMapFilePath, "PENN_SERIES_UID", "PMBB_SERIES_UID")

sectraFilePath = "D:\\data\\pmbb_info\\sectra_syngo_merge_3-1-19_WITHCPT.csv"
accessionToCPT = csv_utils.getDictionaryFromCSV(sectraFilePath, "ACCESSION_NUMBER","CPT")

tagsFilePath = "D:\\test_transfer\\AnonymizedTest\\dicom_tags.csv"
tagsFieldNameList = ["ACCESSION_NUMBER","STUDY_UID","SERIES_UID", "TAGS"]
if not os.path.isfile(tagsFilePath.encode("UTF-8")):
	with open(tagsFilePath,'w') as outputFile:
		writer = csv.DictWriter(outputFile, lineterminator="\n", fieldnames=tagsFieldNameList)
		writer.writeheader()

logFilePath = "D:\\test_transfer\\AnonymizedTest\\log.txt"
if not os.path.isfile(logFilePath.encode("UTF-8")):
	with open(logFilePath, 'w'):
		pass

segmentQuery = os.path.join(os.path.normpath(rootDirPath), "*\\")
segmentDirPathList = glob.glob(segmentQuery)
segmentDirPathList = [token for token in segmentDirPathList if os.path.isdir(token.encode("UTF-8"))]

tagTime = 0
niiTime = 0
timeStart = time.time()
numStudiesProcessed = 0



accessionSet = set()
patientIDSet = set()
studyUIDSet = set()
seriesUIDSet = set()

filePathListTotal = glob.glob(os.path.join(os.path.normpath(rootDirPath), "**"), recursive=True)
timeStart = time.time()
print("parsing")
for fileIndex, filePath in enumerate(filePathListTotal):
	if not os.path.isfile(filePath.encode("UTF-8")):
		continue
	if fileIndex % 1000 == 0:
		print("here")
	dcmFile = pydicom.dcmread(filePath, stop_before_pixels=True, specific_tags=["AccessionNumber","PatientID","StudyInstanceUID","SeriesInstanceUID"])
	accessionSet.add(dcmFile.AccessionNumber)
	patientIDSet.add(dcmFile.PatientID)
	studyUIDSet.add(dcmFile.StudyInstanceUID)
	seriesUIDSet.add(dcmFile.SeriesInstanceUID)

print(time.time() - timeStart)

'''

for segmentIndex, segmentDirPath in enumerate(segmentDirPathList):
	#print("Processing segment %d of %d - %s" % (segmentIndex+1, len(segmentDirPathList), segmentDirPath))
	studyQuery = os.path.join(os.path.normpath(segmentDirPath), "*\\")
	studyDirPathList = glob.glob(studyQuery)
	studyDirPathList = [token for token in studyDirPathList if os.path.isdir(token.encode("UTF-8"))]

	for studyIndex, studyDirPath in enumerate(studyDirPathList):
		if numStudiesProcessed > 0:
			timeEllapsed = time.time() - timeStart
			timePerStudy = timeEllapsed / numStudiesProcessed
			timePer10K = timePerStudy * 10000
			print("%s ellapsed - %s per 10K" % (getTimeString(timeEllapsed), getTimeString(timePer10K)))
			print("Total: %0.2f" % timeEllapsed)
			print("Tag: %0.2f" % tagTime)
			print("NII: %0.2f" % niiTime)
		filePathList = glob.glob(os.path.join(studyDirPath, "*"))
		# Ensure all filepaths are files
		for filePath in filePathList:
			if os.path.isdir(filePath.encode("UTF-8")):
				print("%s is not a file" % filePath)
		# Separate each file by the series that it belongs to
		dcmFileList = []
		for filePath in filePathList:
			try:
				dcmFile = pydicom.dcmread(filePath)
				dcmFileList.append(dcmFile)
			except:
				print(filePath)
				traceback.print_exc()
		if len(dcmFileList) == 0:
			print("%s has no files" % studyDirPath)
			continue
		# Separate by series
		fileMap = {}
		for fileIndex, dcmFile in enumerate(dcmFileList):
			seriesUID = dcmFile.SeriesInstanceUID
			if seriesUID in fileMap:
				fileMap[seriesUID].append(dcmFile)
			else:
				fileMap[seriesUID] = [dcmFile]
		# Get information for each series
		seriesInfoMap = {}
		for seriesUID in fileMap:
			if len(fileMap[seriesUID]) > 1:
				try:
					sortedList = dicom2nifti.common.sort_dicoms(fileMap[seriesUID])
					fileMap[seriesUID] = sortedList
				except:
					print("issue")
			dcmList = fileMap[seriesUID]
			orientationToken = getOrientationForFileList(dcmList)
			sliceThickness = getSliceThicknessForSortedFiles(dcmList, orientationToken)
			sliceThickness = "%0.2f" % sliceThickness if sliceThickness != None else "None"
			numImages = len(dcmList)
			seriesInfoMap[seriesUID] = (orientationToken, sliceThickness, str(numImages))

		# Ensure only one patient in folder
		patientID = dcmFileList[0].PatientID
		accessionNumber = dcmFileList[0].AccessionNumber
		studyUID = dcmFileList[0].StudyInstanceUID
		for fileIndex, dcmFile in enumerate(dcmFileList[1:]):
			if dcmFile.PatientID != patientID:
				print("Non-matching patientID for %s" % studyDirPath)
			if dcmFile.AccessionNumber != accessionNumber:
				print("Non-matching AccessionNumber for %s" % studyDirPath)
			if dcmFile.StudyInstanceUID != studyUID:
				print("Non-matching StudyUID for %s" % studyDirPath)
		# Get anonymzied identifiers
		# Patient ID
		if patientID not in empiToPacketUUID:
			print("%s not in patient mapping" % patientID)
			continue
		packetUUID = empiToPacketUUID[patientID]

		# Accession number
		if accessionNumber not in accessionToPMBB:
			pmbbAccession = str(random.randint(100000000000,999999999999))
			while pmbbAccession in accessionToPMBB.values():
				pmbbAccession = str(random.randint(100000000000,999999999999))
			with open(accessionMapFilePath, 'a') as outputFile:
				writer = csv.DictWriter(outputFile, lineterminator="\n", fieldnames=["PENN_ACCESSION","PMBB_ACCESSION"])
				writer.writerow({"PENN_ACCESSION": accessionNumber, "PMBB_ACCESSION": pmbbAccession})
			accessionToPMBB[accessionNumber] = pmbbAccession
		else:
			pmbbAccession = accessionToPMBB[accessionNumber]

		cptCode = accessionToCPT[accessionNumber]
		# Create output folders
		outputPatientDirPath = os.path.join(outputDirPath, packetUUID)
		outputStudyDirPath = os.path.join(outputPatientDirPath, "%s-%s" %(pmbbAccession, str(cptCode)))
		if not os.path.isdir(outputPatientDirPath.encode("UTF-8")):
			os.mkdir(outputPatientDirPath)
		if not os.path.isdir(outputStudyDirPath.encode("UTF-8")):
			os.mkdir(outputStudyDirPath)
		else:
			print("Study output fodler already exists for %s" % outputStudyDirPath)

		for seriesIndex, seriesUID in enumerate(fileMap):
			if seriesUID not in seriesUIDToPMBB:
				pmbbSeriesUID = str(random.randint(100000000000000,999999999999999))
				while pmbbSeriesUID in seriesUIDToPMBB.values():
					pmbbSeriesUID = str(random.randint(100000000000000,999999999999999))
				with open(seriesUIDMapFilePath, 'a') as outputFile:
					writer = csv.DictWriter(outputFile, lineterminator="\n", fieldnames=["PENN_SERIES_UID","PMBB_SERIES_UID"])
					writer.writerow({"PENN_SERIES_UID": seriesUID, "PMBB_SERIES_UID": pmbbSeriesUID})
				seriesUIDToPMBB[seriesUID] = pmbbSeriesUID
			else:
				pmbbSeriesUID = seriesUIDToPMBB[seriesUID]
			dcmList = fileMap[seriesUID]
			(orientationToken, sliceThickness, numImages) = seriesInfoMap[seriesUID]
			seriesFileName = "%s-%s-%s-%s-%s-%s.nii.gz" % (os.path.basename(os.path.normpath(studyDirPath)), seriesUID, pmbbSeriesUID, orientationToken, sliceThickness, numImages)
			seriesOutputPath = os.path.join(outputStudyDirPath, seriesFileName)

			localStart = time.time()
			# Get tags for series
			tagDict = {}
			for dcmFile in dcmList:
				dcmFile.remove_private_tags()
				subDict = dictify(dcmFile)

				for key in subDict:
					if key not in tagDict:
						keyStr = "{0:04x}-{1:04x}".format(key.group, key.element)
						tagDict[keyStr] = subDict[key]

			with open(tagsFilePath, 'a') as outputFile:
				writer = csv.DictWriter(outputFile, lineterminator="\n", fieldnames=tagsFieldNameList)
				writer.writerow({
					"ACCESSION_NUMBER": accessionNumber,
					"STUDY_UID": studyUID,
					"SERIES_UID": seriesUID,
					"TAGS": json.dumps(tagDict)
					})
			tagTime += time.time() - localStart
			# Try to create nifti file
			localStart = time.time()
			try:
				affine = dicom2nifti.common.create_affine(dcmList)
				data = dicom2nifti.common.get_volume_pixeldata(dcmList)
				nii = nib.Nifti1Image(data, affine)
				nii.to_filename(seriesOutputPath)
			except:
				with open(logFilePath,'a') as outputFile:
					#print("%s - %s\n" % (dcmList[0].StudyInstanceUID, dcmList[0].SeriesInstanceUID))
					outputFile.write("%s - %s\n" % (studyDirPath, dcmList[0].SeriesDescription))
			niiTime += time.time() - localStart
		numStudiesProcessed += 1
			#os.mkdir(seriesOutputPath)
'''
