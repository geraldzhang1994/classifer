import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
import random
from sklearn.metrics import confusion_matrix


spamData = io.loadmat("../data/spam-dataset/spam_data.mat")
trainData = spamData['training_data']
trainLabel = spamData['training_labels']

numTrain = 4000
shuffled = random.sample(range(trainLabel.shape[1]), trainLabel.shape[1])
trainIdx = shuffled[:numTrain]
testIdx = shuffled[numTrain:]



data = []
target = []

for idx in trainIdx:
	data.append(trainData[idx,:])
	target.append(trainLabel[:, idx])

data = np.array(data)
target = np.array(target).ravel()
model = svm.LinearSVC(C = 1)
model.fit(data, target)


numCorrect = 0
for idx in testIdx:
	pred = model.predict(trainData[idx])
	if trainLabel[:, idx] == pred:
		numCorrect += 1

print "Accuracy is %f" % (float(numCorrect)/float(len(testIdx)))


