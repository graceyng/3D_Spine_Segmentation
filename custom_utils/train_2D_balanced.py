import custom_generators
from custom_generators import Generator3D_Classification
import view_classification_models
from keras.optimizers import SGD, RMSprop, Adam
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import multi_gpu_model
import os
import csv

trainDirPath = "M:\\view_classification\\echo_aws_692000\\train"
validDirPath = "M:\\view_classification\\echo_aws_692000\\valid"

trainDirPath = "data/train"
validDirPath = "data/valid"

outputDirPath = "output/view_class/run_1"

numFramesPerStack = 1
batchSize = 8
inputShape=(224,224)

trainFileIDList, trainFileIDToPath, trainFileIDToLabel, trainNumClasses, labelToViewName = custom_generators.getFileMappingForDir(trainDirPath, numFramesPerStack, sepToken="_")
validFileIDList, validFileIDToPath, validFileIDToLabel, validNumClasses, labelToViewName = custom_generators.getFileMappingForDir(validDirPath, numFramesPerStack, sepToken="_")

trainGenerator = Generator2DClassifier(trainFileIDList, trainFileIDToPath, trainFileIDToLabel, batchSize=batchSize, dim=(224, 224),
								numFramesPerStack=numFramesPerStack, nChannels=1, nClasses=trainNumClasses, shuffle=True,
								sepToken="_", zoomRange=(0.9, 1.1), rotationRange=30, widthShiftRange=0.1, heightShiftRange=0.1)
validGenerator = Generator2DClassifier(validFileIDList, validFileIDToPath, validFileIDToLabel, batchSize=batchSize, dim=(224, 224),
								numFramesPerStack=numFramesPerStack, nChannels=1, nClasses=validNumClasses, shuffle=True,
								sepToken="_", zoomRange=(0.9, 1.1), rotationRange=30, widthShiftRange=0.1, heightShiftRange=0.1)


#==================================================================
# Logging
#==================================================================
fieldNameList = ["epoch","training_loss","validation_loss"]
for label in labelToViewName:
	fieldNameList.append("%s_training" % labelToViewName[label])
	fieldNameList.append("%s_validation" % labelToViewName[label])

class WeightsRecorder(keras.callbacks.Callback):
	def __init__(self, progressFilePath):
		super(WeightsRecorder, self).__init__()
		self.progressFilePath = progressFilePath

	def on_epoch_end(self, epoch, logs=None):
		outputRow["epoch"] = logs["epoch"] - 1
		outputRow["training_loss"] = logs["loss"]
		outputRow["validation_loss"] = logs["val_loss"]
		for label in range(0, trainNumClasses):
			if label == 0:
				trainToken = "class_acc"
				validToken = "val_class_acc"
			else:
				trainToken = "class_acc_%d" % label
				validToken = "val_class_acc_%d" % label
			outputRow["%s_training" % labelToViewName[label]] = logs[trainToken]
			outputRow["%s_validation" % labelToViewName[label]] = logs[validToken]

		with open(progressFilePath, "a") as outputFile: 
			writer = csv.DictWriter(outputFile, lineterminator='\n', fieldnames=fieldNameList)
			writer.writerow(outputRow)


progressFilePath = os.path.join(outputDirPath, "training_progress.csv")
if not os.path.isfile(progressFilePath):
	with open(progressFilePath, "w") as outputFile: 
		writer = csv.DictWriter(outputFile, lineterminator='\n', fieldnames=fieldNameList)
		writer.writeheader()

recorder = WeightsRecorder(progressFilePath)
weight_saver = ModelCheckpoint('model.{epoch:02d}-{val_loss:.4f}.h5',save_best_only=False, save_weights_only=False)
callbackList = [recorder, weight_saver]
#==================================================================
# Metrics
#==================================================================
def single_class_accuracy(interesting_class_id):
	def class_acc(y_true, y_pred):
		class_id_true = K.argmax(y_true, axis=-1)
		class_id_preds = K.argmax(y_pred, axis=-1)
		# Replace class_id_preds with class_id_true for recall here
		accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
		class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
		class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
		return class_acc
	return class_acc

metricList = [single_class_accuracy(index) for index in range(0, trainNumClasses)]

#==================================================================
# Load model and start training
#==================================================================
model = custom_models.getVGGModel(trainNumClasses, inputShape)

model.compile(optimizer=Adam(lr=1e-5, decay=1e-8), loss=keras.losses.categorical_crossentropy)
model.load_weights("weights.37-1.28.h5")
#weight_saver = ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5',save_best_only=True, save_weights_only=False)
model.fit_generator(generator=trainGenerator,
					validation_data=validGenerator,
					validation_steps=1, steps_per_epoch=200, epochs=200, callbacks=callbackList)




