import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
import random
from sklearn.metrics import confusion_matrix
from skimage.feature import hog


# load the train data from a mat file
trainingData = io.loadmat("../data/digit-dataset/train.mat")
trainLabels = trainingData['train_labels']
trainImages = trainingData['train_images']
numData = len(trainLabels)
numTrain = 10000
fold = 10

# a list of possible C values
CValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 5, 8, 10, 50, 100, 1000]

# the training and test data points
trainAndTest = random.sample(range(numData), 10000);
train = trainAndTest[:numTrain/fold]
test = trainAndTest[numTrain/fold:]

for cValue in CValues:


	# construct the data matrix and the target array
	data = []
	target = np.array([], int)
	for idx in train:
		target = np.append(target, trainLabels[idx])
		img = trainImages[0:28, 0:28, idx]
		fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(4, 4))
		features = fd
		data.append(features)
	data = np.array(data)


	model = svm.LinearSVC(C = cValue)
	model.fit(data, target)


	numCorrect = 0
	y_test = []
	y_pred = []
	for idx in test:
		img = trainImages[0:28, 0:28, idx]
		fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(4, 4))
		features = fd
		pred = model.predict(features)[0]
		y_test.append(trainLabels[idx])
		y_pred.append(pred)

		if pred == trainLabels[idx][0]:
			numCorrect = numCorrect + 1
	y_pred = np.array(y_pred)
	y_test = np.array(y_test)

	print "the percentage of correctness is %f when the C Value is %f" % \
		((float(numCorrect)/float(len(test))), cValue)