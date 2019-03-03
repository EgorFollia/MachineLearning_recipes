import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


data = datasets.load_iris()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)

classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)

prediction = classifier.predict(X_test)

print(accuracy_score(Y_test, prediction)) # over 0.93 - 0.94
 
