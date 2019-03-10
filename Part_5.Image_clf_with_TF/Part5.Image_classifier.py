from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import datasets

# load dateset and splitting it
data = datasets.load_iris()
X = data.data
y = data.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = .2, random_state = 42)

# building 3 DNN with 10, 20, 10 units
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
clf = learn.DNNClassifier(feature_columns = feature_columns, hidden_units = [10, 20, 10], n_classes = 3)

# fitting and prediction
clf.fit(train_X, train_y, steps = 200)
score = metrics.accuracy_score(test_y, clf.predict(test_X))

print('Accuracy: ', score)
