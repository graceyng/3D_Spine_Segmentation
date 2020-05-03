import data_utils

'''
imgDirPathList = ["D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\test\\images\\images",
				"D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\train\\images\\images"
				"D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\valid\\images\\images"
]
maskDirPathList = ["D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\test\\masks\\masks",
				"D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\train\\masks\\masks"
				"D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\valid\\masks\\masks"
]
'''
imgDirPathList = ["D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\train\\images\\images",
				"D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\valid\\images\\images"
]
maskDirPathList = ["D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\train\\masks\\masks",
				"D:\\data\\pbp\\bravo_extract\\bravo\\bravo\\pet\\valid\\masks\\masks"
]
outputDirPath = "D:\\data\\pbp\\bravo_extract\\data_valid"
data_utils.createOverlayDir(imgDirPathList, maskDirPathList, outputDirPath)
