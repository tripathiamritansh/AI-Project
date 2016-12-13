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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


image_data = datasets.fetch_mldata('MNIST Original') # Get the MNIST dataset.

basic_x = image_data.data
basic_y = image_data.target # Separate images from their final classification. 

min_max_scaler = preprocessing.MinMaxScaler() # Create the MinMax object.
basic_x = min_max_scaler.fit_transform(basic_x.astype(float)) # Scale pixel intensities only.

arr_acc = []
arr_hog_acc = []

train_x_main, test_x, train_y_main, test_y = cross_validation.train_test_split(basic_x, basic_y, test_size = 0.1, random_state = 0) # Split training/test.

#hogFeatures_test = []
#for feature in test_x:
#	f = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualise=False)
#	hogFeatures_test.append(f)

print "init len ", len(train_x_main)
for i in range(10):
	m=int(len(train_x_main)*(i+1.0)/10)
	train_x = train_x_main[0:m]
	print "len at ", i, " iteration : ", len(train_x)
	train_y = train_y_main[0:m]
	clf_sgd = SGDClassifier()
	clf_sgd.fit(train_x, train_y)
	y_pred_sgd = clf_sgd.predict(test_x)
	acc_sgd = accuracy_score(test_y, y_pred_sgd)
	arr_acc.append(acc_sgd)
#	hogFeatures_train = []
#	for feature in train_x:
#		f = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualise=False)
#		hogFeatures_train.append(f)
#	clf_sgd_hog = SGDClassifier()
#	clf_sgd_hog.fit(hogFeatures_train, train_y)
#	y_pred_sgd_hog = clf_sgd_hog.predict(hogFeatures_test)
#	acc_sgd_hog = accuracy_score(test_y, y_pred_sgd_hog)
#	arr_hog_acc.append(acc_sgd_hog)

print arr_acc
arr_x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.plot(arr_x,arr_acc,'b-')#,arr_x,arr_hog_acc,'r-')
plt.ylabel('accuracy')
plt.xlabel('ratio of training data')
blue_patch = mpatches.Patch(color='blue', label='SGD w/o hog')
#red_patch = mpatches.Patch(color='red', label='SGD w/ hog')
#plt.legend(handles=[blue_patch])#, red_patch], loc=4)
plt.savefig('sgd_mnist.png')
plt.show()


