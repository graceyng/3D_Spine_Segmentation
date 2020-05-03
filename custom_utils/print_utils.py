
def printList(myList):
	print("\n".join(myList))
def printParallelLists(list1, list2, delimiter=" - "):
	if len(list1) != len(list2):
		raise ValueError("Lists are of different sizes: %d and %d" % (len(list1), len(list2)))
	for index in range(0, len(list1)):
		print("%s%s%s" % (list1[index], delimiter, list2[index]))
def getTimeString(seconds, includeDays=False):
	remaining = float(seconds)
	if includeDays == True:
		days = seconds / (24 * 3600)
		remaining %= (24 * 3600)
		hours = remaining / 3600
		remaining %= 3600
		minutes = remaining / 60
		remaining %= 60
		seconds = remaining
		return "%d:%02d:%02d:%02d" % (days, hours, minutes, seconds)
	else:
		hours = remaining / 3600
		remaining %= 3600
		minutes = remaining / 60
		remaining %= 60
		seconds = remaining
		return "%02d:%02d:%02d" % (hours, minutes, seconds)