import numpy as np
import glob
import os
import shutil
import re
import random
import pydicom
import nibabel as nib
import cv2
import json


dirPath = '/Users/graceng/Documents/Med_School/Research/Radiology/all_mri_cleaned_bbs/'
scanRegExFilter = r'_L$' #None
sliceFileExt = '.dcm'
segFile = 'seg.nii.gz'
segLabelKey = {'disc': 1.}
collateSegType = 'vert' #'disc'
createMaskDir = None #'collateVertMasks' #'collateDiscMasks' #None
outputFile = None #'/Users/graceng/Documents/Med_School/Research/Radiology/043020_Metadata.json'


#Parameters for splitting training vs. validation sets
k_folds = 5
k_idx = 1
seed = 1
batch_size = 5 #1 # 32


def getFolderNamesFromDir(dirPath, namesOnly=True, regExFilter=None):
    folderPaths = glob.glob(os.path.join(os.path.normpath(dirPath), "*") + "/")
    if not namesOnly:
        return folderPaths
    folderNames = sorted([os.path.basename(os.path.normpath(path)) for path in folderPaths])
    if regExFilter is not None:
        folderNames = [str for str in folderNames if re.search(regExFilter, str)]
    return folderNames

def getFileNamesFromDir(dirPath, namesOnly=True, fileExt=None):
    # Note: make sure that the filenames are in a format such that the sorting function will return the desired order
    # (e.g. name using 0001, 0002, etc. instead of 1, 10, 11, 2, etc.
    if fileExt == None:
        filePathList = glob.glob(os.path.join(os.path.normpath(dirPath), "*"))
    else:
        filePathList = glob.glob(os.path.join(os.path.normpath(dirPath), "*"+fileExt))
    for filePath in filePathList:
        if not os.path.isfile(filePath):
            raise ValueError("%s is not a file" % filePath)
    if not namesOnly:
        return filePathList
    fileNameList = sorted([os.path.basename(os.path.normpath(path)) for path in filePathList])
    return fileNameList


#Feed a list of filepaths to the scan directories into the generator. Need to split dataset randomly into training vs. validation sets
scanNames = getFolderNamesFromDir(dirPath, regExFilter=scanRegExFilter)
total_images = len(scanNames)
total_valid_images = int(np.floor(total_images / k_folds))
total_train_images = total_images - total_valid_images


max_frames = 0
max_height = 0
max_width = 0
min_height = 1000
min_width = 1000
for scanName in scanNames:
    sliceNames = getFileNamesFromDir(dirPath+scanName, fileExt=sliceFileExt)
    num_frames = len(sliceNames)
    if num_frames > max_frames:
        max_frames = num_frames
    for sliceName in sliceNames:
        imgFilePath = dirPath+scanName+'/'+sliceName
        dataset = pydicom.dcmread(imgFilePath)
        Img_slice = dataset.pixel_array
        (height, width) = Img_slice.shape
        if height > max_height:
            max_height = height
        if height < min_height:
            min_height = height
        if width > max_width:
            max_width = width
        if width < min_width:
            min_width = width
    if createMaskDir is not None:
        maskDirPath = dirPath+scanName+'/'+createMaskDir
        if os.path.exists(maskDirPath):
            shutil.rmtree(maskDirPath)
        os.mkdir(maskDirPath)
        i = nib.load(dirPath+scanName+'/'+segFile)
        i = nib.as_closest_canonical(i)
        i = i.get_fdata()
        idxStrWidth = len(str(len(sliceNames)))+2
        for idx in range(len(sliceNames)):
            c = i[idx,]
            c = np.rot90(c)
            c = np.fliplr(c)
            #create mask for only vertebrae or disc segmentations
            if collateSegType == 'vert':
                c = (c > segLabelKey['disc']).astype('uint8') * 255
            elif collateSegType == 'disc':
                c = (c == segLabelKey['disc']).astype('uint8') * 255
            else:
                raise Exception('collateSegType must be either "vert" or "disc".')
            cv2.imwrite(maskDirPath+'/{:0{}d}_slice_all_'.format(idx+1, idxStrWidth)+collateSegType+'.png', c)


# NUM FRAMES PER STACK IS THE TOTAL NUMBER OF SLICES
numFramesPerStack = max_frames
# Note: within our MRI dataset, min number of frames is 11 and max number is 20

random.seed(seed)
random.shuffle(scanNames)

scanList_train = scanNames[:total_train_images-1]
scanList_valid = scanNames[total_train_images:]

output_dict = {'scanList_train': scanList_train, 'scanList_valid': scanList_valid, 'dim': (max_height, max_width),
               'numFramesPerStack': numFramesPerStack, 'dirPath': dirPath, 'sliceFileExt': sliceFileExt,
               'batch_size': batch_size, 'k_folds': k_folds, 'k_idx': k_idx, 'seed': seed}
if outputFile is not None:
    with open(outputFile, 'w') as outfile:
        json.dump(output_dict, outfile)