import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import preprocessing
from skimage.feature import hog
from nolearn.dbn import DBN
import timeit
import time

image_data = datasets.fetch_mldata('MNIST Original') # Get the MNIST dataset.

basic_x = image_data.data
basic_y = image_data.target # Separate images from their final classification. 

min_max_scaler = preprocessing.MinMaxScaler() # Create the MinMax object.
basic_x = min_max_scaler.fit_transform(basic_x.astype(float)) # Scale pixel intensities only.

train_x, test_x, train_y, test_y = cross_validation.train_test_split(basic_x, basic_y, test_size = 0.2, random_state = 0) # Split training/test.

hogFeatures_train = []
for feature in train_x:
	f = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualise=False)
	hogFeatures_train.append(f)

print len(hogFeatures_train)

hogFeatures_test = []
for feature in test_x:
	f = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualise=False)
	hogFeatures_test.append(f)


clf_svm = LinearSVC()
start = time.time()
clf_svm.fit(hogFeatures_train, train_y)
finish = time.time()
y_pred_svm = clf_svm.predict(hogFeatures_test)
acc_svm = accuracy_score(test_y, y_pred_svm)
print "Linear SVM accuracy: ",acc_svm
print "time : ", finish - start