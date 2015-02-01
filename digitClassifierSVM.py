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
numData = len(trainLabels)
trainImages = trainingData['train_images']


if False:
	# a list of number of training examples
	numTrainExmaple = [50000,] # 200, 500, 1000, 2000, 5000, 10000]


	for numTrain in numTrainExmaple:
		trainAndTest = random.sample(range(numData), numTrain + 10000);
		train = trainAndTest[:numTrain]
		test = trainAndTest[numTrain:]
		# construct the data matrix and the target array
		data = []
		target = np.array([], int)
		for idx in train:
			target = np.append(target, trainLabels[idx])
			img = trainImages[0:28, 0:28, idx]
			fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
	                    cells_per_block=(4, 4))
			features = fd# np.append(img.ravel(), fd)
			data.append(features)
		data = np.array(data)
		#print data.shape
		#print target.shape

		model = svm.LinearSVC(C = 3)
		model.fit(data, target)
		# print model

		numCorrect = 0
		y_test = []
		y_pred = []
		for idx in test:
			img = trainImages[0:28, 0:28, idx]
			fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
	                    cells_per_block=(4, 4))
			features = fd# np.append(img.ravel(), fd)
			pred = model.predict(features)[0]
			y_test.append(trainLabels[idx])
			y_pred.append(pred)

			if pred == trainLabels[idx][0]:
				numCorrect = numCorrect + 1
		y_pred = np.array(y_pred)
		y_test = np.array(y_test)

		cm = confusion_matrix(y_test, y_pred)
		print cm
		plt.matshow(cm)
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()


		print "the percentage of correctness is %f when the number of training data is %d" % \
			((float(numCorrect)/float(len(test))), numTrain)


testData = io.loadmat("../data/digit-dataset/test.mat")
testImages = testData['test_images']
numTest = testImages.shape[2]

data = []
target = np.array([], int)
for idx in range(numData):
	target = np.append(target, trainLabels[idx])
	img = trainImages[0:28, 0:28, idx]
	fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
                cells_per_block=(4, 4))
	features = fd# np.append(img.ravel(), fd)
	data.append(features)
data = np.array(data)
#print data.shape
#print target.shape

model = svm.LinearSVC(C = 3)
model.fit(data, target)

f = open('result.csv', 'w')
f.write('Id,Category\n')
for idx in range(numTest):
	img = testImages[0:28, 0:28, idx]
	fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
                cells_per_block=(4, 4))
	features = fd# np.append(img.ravel(), fd)
	pred = model.predict(features)[0]
	f.write(str(idx + 1) + ',' + str(pred) + '\n')
	
f.close()