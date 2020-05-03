import csv

def getColumnFromCSV(csvFilePath, columnName, lower=False):
	with open(csvFilePath, 'r') as file:
		reader = csv.DictReader(file)
		myList = []
		for row in reader:
			if lower:
				myList.append(row[columnName].strip().lower())
			else:
				myList.append(row[columnName].strip())
	return myList

def getDictionaryFromCSV(csvFilePath, keyColumnName, valueColumnName, lower=False):
	keys = []
	values = []
	with open(csvFilePath, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			if lower:
				keys.append(row[keyColumnName].strip().lower())
				values.append(row[valueColumnName].strip().lower())
			else:
				keys.append(row[keyColumnName].strip())
				values.append(row[valueColumnName].strip())
	return dict(zip(keys, values))