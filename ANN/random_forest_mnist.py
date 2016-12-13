import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import preprocessing
from nolearn.dbn import DBN
import timeit
import time

image_data = datasets.fetch_mldata('MNIST Original') # Get the MNIST dataset.

basic_x = image_data.data
basic_y = image_data.target # Separate images from their final classification. 

min_max_scaler = preprocessing.MinMaxScaler() # Create the MinMax object.
basic_x = min_max_scaler.fit_transform(basic_x.astype(float)) # Scale pixel intensities only.

train_x, test_x, train_y, test_y = cross_validation.train_test_split(basic_x, basic_y, test_size = 0.2, random_state = 0) # Split training/test.

clf_rf = RandomForestClassifier()
start = time.time()
clf_rf.fit(train_x, train_y)
finish = time.time()
y_pred_rf = clf_rf.predict(test_x)
acc_rf = accuracy_score(test_y, y_pred_rf)
print "random forest accuracy: ",acc_rf
print "time  : ", finish - start