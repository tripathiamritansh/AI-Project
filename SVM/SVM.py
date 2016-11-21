from sklearn import svm
import csv
import matplotlib as plt

train_data=open('train.csv')
reader= csv.reader(train_data,delimiter=",")
data = list(reader)
row_count =len(data)

