from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os

'''
Let feautres are Weight and Texture
As classifier works only with real-values, lets denote 0 - bumpy and 1 - smooth
features = [[140, 1],
            [130, 1],
            [150, 0],
            [170, 0]]
For labels we set 0 - apple and 1 - orange
labels = [0, 0, 1, 1]
'''

df = pd.read_csv(os.getcwd() + '/fruits.csv')
features = df.values[:, 1:3]
labels = df.values[:,3]

clf = DecisionTreeClassifier()
clf = clf.fit(features, labels)

w = int(input('Enter weight '))
l = int(input('Enter label '))

print('I predict ... ', end = '')

if (clf.predict([[w, l]]) == np.array([1])):
    print('orange')
else:
    print('apple')
