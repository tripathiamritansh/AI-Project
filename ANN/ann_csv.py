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

train = pd.read_csv("../train.csv")
features = train.columns[1:]
X = train[features]
y = train['label']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X/255.,y,test_size=0.1,random_state=0)


clf_nn = DBN([X_train.shape[1], 300, 10],learn_rates=0.3,learn_rate_decays=0.9,epochs=15)
training_x = X_train.as_matrix()
training_y = y_train.as_matrix()
testing_x = X_test.as_matrix()
testing_y = y_test.as_matrix()
clf_nn.fit(training_x, training_y)
acc_nn = clf_nn.score(testing_x,testing_y)
print "neural network accuracy: ",acc_nn