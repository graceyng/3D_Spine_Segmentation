
import os
import glob

#----------------------------------------------------
def getDirNamesFromDir(dirPath):
	return getDirsFromDir(dirPath, namesOnly=True)
def getDirPathsFromDir(dirPath):
	return getDirsFromDir(dirPath, namesOnly=False)
def getDirsFromDir(dirPath, namesOnly=True):
	dirPaths = glob.glob(os.path.join(os.path.normpath(dirPath), "*") + "/")
	if not namesOnly:
		return dirPaths
	dirNames = [os.path.basename(os.path.normpath(path)) for path in dirPaths]
	return dirNames
#----------------------------------------------------
def getFileNamesFromDir(dirPath, ext=None):
	return getFilesFromDir(dirPath, namesOnly=True, ext=ext)
def getFilePathsFromDir(dirPath, ext=None):
	return getFilesFromDir(dirPath, namesOnly=False, ext=ext)
def getFilesFromDir(dirPath, namesOnly=True, ext=None):
	if ext == None:
		dirPaths = glob.glob(os.path.join(os.path.normpath(dirPath), "*"))
	else:
		queryPath = os.path.join(os.path.normpath(dirPath), "*%s" % ext)
		dirPaths = glob.glob(queryPath)
	if not namesOnly:
		return dirPaths
	dirNames = [os.path.basename(os.path.normpath(path)) for path in dirPaths]
	return dirNames
def getRecursiveFilePathsFromDir(dirPath, ext=None):
	filePathList = []
	for dirpath, dirnames, filenames in os.walk(dirPath):
		if ext == None:
			filePathList += filenames
		else:
			filePathList += [f for f in filenames if f.endswith(ext)]
#----------------------------------------------------

def getNumFilesInDir(dirPath):
	return len(getFilesFromDir(dirPath, namesOnly=True)) 
def getDirNameFromDirPath(dirPath):
	return os.path.basename(os.path.normpath(dirPath))

def getUniqueDirPath(rootPath, dirName):
	dirNameStart = dirName
	dirNameFinal = dirName
	index = 1
	while os.path.isdir(os.path.join(rootPath, dirNameFinal)):
		dirNameFinal = "%s_%d" % (dirNameStart, index)
		index += 1
	return os.path.join(rootPath, dirNameFinal), dirNameFinal
def getUniqueFilePath(rootDirPath, fileName):
	fileNameBase, ext = os.path.splitext(fileName)
	
	fileNameBaseStart = fileNameBase
	fileNameBaseFinal = fileNameBase
	index = 1
	while os.path.isfile(os.path.join(rootDirPath, "%s%s" % (fileNameBaseFinal, ext))):
		fileNameBaseFinal = "%s_%d" % (fileNameBaseStart, index)
		index += 1
	fileName = "%s%s" % (fileNameBaseFinal, ext)
	filePath = os.path.join(rootDirPath, fileName)
	return filePath, fileName















