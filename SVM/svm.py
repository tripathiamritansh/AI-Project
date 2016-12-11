import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC


#Fetching the datasets
dataset = datasets.fetch_mldata("MNIST Original")
#dataset = datasets.load_digits()

for key, _ in dataset.iteritems():
	print key

training_data = dataset.data[:70000]
training_target = dataset.target[:70000]

testing_data = dataset.data[-40000:]
testing_target = dataset.target[-40000:]


#creating features and labels
featureSet = np.array(training_data, 'int16')
labelSet = np.array(training_target, 'int')

#Calculating Hog features

hogFeatures = []
for feature in featureSet:
	f = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	hogFeatures.append(f)
hog_features=np.array(hogFeatures, 'float64')

#instantiating classifier
linear_svm=LinearSVC()

#Training
linear_svm.fit(hog_features,labelSet)

featureSet_testing=np.array(testing_data, 'int16')

hogFeatures_testing = []
for feature in featureSet_testing:
	f_testing = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	hogFeatures_testing.append(f_testing)
hog_features_testing=np.array(hogFeatures_testing, 'float64')
#Testing

s = 0.0
print hogFeatures_testing
"""
for i, _ in enumerate(testing_data):
	if linear_svm.predict(hog_features_testing)[0] == testing_target[i]:
		s += 1

print "accuracy = ", s/len(testing_data)
"""