import sys
sys.path.insert(0,"..\\")
import nibabel as nib
from custom_utils import display
from custom_utils import dcm_utils
from custom_utils import img_utils
import numpy as np
import time

def getLungMask(stack):
	level = -300
	width = 1
	# Treshold for lungs
	stackMask = dcm_utils.thresholdImage(stack, level, width) == 255

	# Fill in any holes
	stackMaskFilled = img_utils.imfill_stack(stackMask.copy())

	# Get largest components which is the body
	bodyMask = img_utils.getLargestCC(stackMaskFilled)
	# Invert stackMask so that lungs are white (255)
	stackMask = np.invert(stackMask)
	# Exclude regions outside the body
	stackMask[bodyMask == False] = False
	lungMask = img_utils.getLargestCC(stackMask,numObjects=2)
	lungMask = img_utils.imfill_stack(lungMask)
	return lungMask

# Works for abdominal CT and CT/Pel
def checkIfAbdomenCTIsContrast(stack):
	lungMask = getLungMask(stack)
	areaList = []
	for zIndex in range(0, lungMask.shape[2]):
		sliceMask = lungMask[:,:,zIndex]
		areaList.append(np.sum(sliceMask))
	maxZ = np.argmax(areaList)
	# Get five greatest-area slices centered on
	# maximum area slice
	indsToProcess = [maxZ]
	cursorBelow = maxZ - 1
	cursorAbove = maxZ + 1
	for numSlices in range(0, 5):
		if cursorBelow < 0:
			indsToProcess.append(cursorAbove)
			continue
		elif cursorAbove >= len(areaList):
			indsToProcess.append(cursorBelow)
			continue
		if areaList[cursorBelow] > areaList[cursorAbove]:
			indsToProcess.append(cursorBelow)
			cursorBelow -= 1
		else:
			indsToProcess.append(cursorAbove)
			cursorAbove += 1
	print(indsToProcess)
	# Add up HU of those slices
	totalHU = 0
	for zIndex in indsToProcess:
		sliceLungMask = lungMask[:,:,zIndex]
		sliceStack = stack[:,:,zIndex]
		sliceStack[sliceStack < 50] = 0
		totalHU += np.sum(sliceStack[sliceLungMask])
	return totalHU


if __name__ == "__main__":
	dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_iv_only\\10201387\\CT AbdomenPel 5.0 I40f 3"
	dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_iv_only\\21068367\\CT ABDPELVIS 5MM"
	dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_iv_only\\24734178\\CT 5MM AP W"
	
	#dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_no_iv_no_po\\15285736\\Abd-Pel  5.0  I40f  3"
	#dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_no_iv_with_po\\51822811\\Abdomen-Pel  5.0  Br40  3"
	dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_no_iv_with_po\\29341946\\CT AbdPel 5.0 B30f ST"
	#dcmDirPath = "D:\\data\\contrast_classifier\\abd_pel_no_iv_with_po\\45020633\\CT abd_pelvis 5.0 B30f"
	imgStack = dcm_utils.getRawStackFromDCMDirPath(dcmDirPath, query="*")
	startTime = time.time()
	# File Paths
	#imgFilePath = "81419617_b.nii.gz"
	#maskFilePath = "81419617_b_MASK.nii.gz"

	# Load original image
	#img = nib.load(imgFilePath)
	#imgStack = img.get_fdata()
	# Load aorta mask
	#img = nib.load(maskFilePath)
	#aortaMask = img.get_fdata()
	totalHU = checkIfAbdomenCTIsContrast(imgStack)
	print("Ellapsed: %0.2f" % (time.time() - startTime))
	print(totalHU)
	# Load lung mask
	#lungMask = getLungMask(imgStack)

