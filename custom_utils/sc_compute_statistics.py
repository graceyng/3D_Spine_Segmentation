import sys
sys.path.insert(0, "C:\\projects\\net_learn")

from custom_utils import csv_utils

csvFilePath = "D:\\data\\pbp\\segment_project\\subject_data.csv"

trainSetList = csv_utils.getColumnFromCSV(csvFilePath, "train")
trainSetList = [int(item) for item in trainSetList]

patternSetList = csv_utils.getColumnFromCSV(csvFilePath, "FDG_PATTERN")
patternSetList = [int(item) for item in patternSetList]

idList = csv_utils.getColumnFromCSV(csvFilePath, "FDG_ID")
idList = [int(item) for item in idList]