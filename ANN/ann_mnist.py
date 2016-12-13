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


clf_nn = DBN([train_x.shape[1], 300, 10],learn_rates=0.3,learn_rate_decays=0.9,epochs=15)
training_x = train_x#.as_matrix()
training_y = train_y#.as_matrix()
testing_x = test_x#.as_matrix()
testing_y = test_y#.as_matrix()
start = time.time()
clf_nn.fit(training_x, training_y)
finish = time.time()
acc_nn = clf_nn.score(testing_x,testing_y)
print "neural network accuracy: ",acc_nn
print "time : ", finish - start