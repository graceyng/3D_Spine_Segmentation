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


# Note: for the paths, be sure to have a '/' at the end
dirPath = '/Volumes/Storage/Radiology_Data/mri_all_files/'
# dirPath = '/Users/graceng/Documents/Med_School/Research/Radiology/all_mri_cleaned_bbs/'

# duplicatesPath: a directory that contains scans that were previously processed. If a scan name is found in both the
# dirPath and the duplicatesPath (meaning that the scan has already been processed), either skip the processing of this
# scan in dirPath, or remove the scan directory within dirPath
duplicatesPath = '/Users/graceng/Documents/Med_School/Research/Radiology/all_mri_cleaned_bbs/' #None
duplicatesOutputFile = '/Users/graceng/Documents/Med_School/Research/Radiology/043020_Metadata.json' #None
handleDuplicates = 'remove' #'skip'
scanRegExFilter = None #r'_L$'
sliceFileExt = '.dcm'
segFile = 'seg.nii.gz'
segLabelKey = {'disc': 1.}
collateSegType = 'vert' #'disc'
createMaskDir = None #'collateVertMasks' #'collateDiscMasks' #None

# outputHandleDuplicates: if 'combine', then both scans from dirPath and duplicatesPath are included in the outputFile
# (without duplication). If 'currDirOnly', then only scans from dirPath (regardless of whether they also occur in duplicatesPath)
# are included in the outputFile.
outputHandleDuplicates = 'combine' #'currDirOnly'
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
max_frames = 0
max_height = 0
max_width = 0

if duplicatesPath is not None:
    duplicateNames = getFolderNamesFromDir(duplicatesPath)
    if outputHandleDuplicates == 'combine':
        if duplicatesOutputFile is None:
            raise Exception('In order to create a combined output file, duplicatesOutputFile cannot be none.')
        else:
            metadataVars = ['scanList_train', 'scanList_valid', 'dim', 'numFramesPerStack', 'dirPath', 'sliceFileExt',
                            'batch_size', 'k_folds', 'k_idx', 'seed']
            f = open(duplicatesOutputFile, 'r')
            duplicatesMetadata = json.load(f)
            f.close()
            max_frames = duplicatesMetadata['numFramesPerStack']
            max_height = duplicatesMetadata['dim'][0]
            max_width = duplicatesMetadata['dim'][1]

for scanName in scanNames:
    if duplicatesPath is not None:
        if scanName in duplicateNames:
            if handleDuplicates == 'skip':
                break
            elif handleDuplicates == 'remove':
                shutil.rmtree(dirPath+scanName)
            else:
                raise Exception('handleDuplicates must be either "skip" or "remove".')
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
        if width > max_width:
            max_width = width
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

if duplicatesPath is not None:
    if outputHandleDuplicates == 'combine':
        if numFramesPerStack != duplicatesMetadata['numFramesPerStack']:
            print('FYI: numFramesPerStack from the set of scans contained in dirPath exceeds that of the set of scans in '
                  'duplicatesPath.')
        if max_height != duplicatesMetadata['dim'][0]:
            print('FYI: max_height (metadata["dim"][0]) from the set of scans contained in dirPath exceeds that of the '
                  'set of scans in duplicatesPath.')
        if max_width != duplicatesMetadata['dim'][1]:
            print('FYI: max_width (metadata["dim"][1]) from the set of scans contained in dirPath exceeds that of the '
                  'set of scans in duplicatesPath.')
        outputScanNames = list(set(scanNames) | set(duplicateNames))
    elif outputHandleDuplicates == 'currDirOnly':
        outputScanNames = scanNames
    else:
        raise Exception('outputHandleDuplicates must be either "combine" or "currDirOnly."')
else:
    outputScanNames = scanNames

total_images = len(outputScanNames)
total_valid_images = int(np.floor(total_images / k_folds))
total_train_images = total_images - total_valid_images
scanList_train = outputScanNames[:total_train_images-1]
scanList_valid = outputScanNames[total_train_images:]

output_dict = {'scanList_train': scanList_train, 'scanList_valid': scanList_valid, 'dim': (max_height, max_width),
               'numFramesPerStack': numFramesPerStack, 'dirPath': dirPath, 'sliceFileExt': sliceFileExt,
               'batch_size': batch_size, 'k_folds': k_folds, 'k_idx': k_idx, 'seed': seed}
if outputFile is not None:
    with open(outputFile, 'w') as outfile:
        json.dump(output_dict, outfile)