# Predicion the flower by the label & visualize tree
import numpy as np
import graphviz, os
import pydot
from sklearn.externals.six import StringIO
from sklearn.datasets import load_iris
from sklearn import tree

# 1 -- import data flower dataset
data = load_iris()
print(data.feature_names)
print(data.target_names)

print(data.data[0])
print(data.target[0])

for i in range(len(data.target)):
    print(i, 'features :', *data.data[i], 'label :', data.target[i])


# 2 -- train a classifier
# filtering data
testing_ind = [0, 50, 100]

# training data
target_train = np.delete(data.target, testing_ind)
data_train = np.delete(data.data, testing_ind, axis = 0)

# testing data
target_test = data.target[testing_ind]
data_test = data.data[testing_ind]

clf = tree.DecisionTreeClassifier()
clf.fit(data_train, target_train)


# 3 -- perdiction
print(target_test)
print(clf.predict(data_test))


# 4 -- visualization using viz code scikit-learn visualization
dot_data = StringIO()
dot_date = tree.export_graphviz(clf, out_file = dot_data,
                     feature_names = data.feature_names,
                     class_names = data.target_names,
                     filled = True, rounded = True,
                     impurity = False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf(os.getcwd()  + "Iris_tree.pdf")
