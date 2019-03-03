# Creation of learning algorithm and dividing data into test and train
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# loading our iris data
data = datasets.load_iris()

# get data from dataset
X = data.data
Y = data.target

# divide it into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)
# 150 examples -> 75 to test and 75 to train

# made classifier and train it using our prepared data
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

# let's predict
prediction = classifier.predict(X_test)

# check accuracy
print(accuracy_score(Y_test, prediction)) # over 0.98 - 0.97
