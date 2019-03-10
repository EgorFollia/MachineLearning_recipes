from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def dstnc(a, b):
    return distance.euclidean(a, b)

class self_kNN():
    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def predict(self, test_X):
        predictions = []

        for row in test_X:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = dstnc(row, self.train_X[0])
        best_ind = 0

        for ind in range(1, len(self.train_X)):
            dist = dstnc(row, self.train_X[ind])
            if dist < best_dist:
                best_dist = dist
                best_ind = ind

        return self.train_y[best_ind]

data = datasets.load_iris()
X = data.data
y = data.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = .5)

clf = self_kNN()
clf.fit(train_X, train_y)

predictions = clf.predict(test_X)
print(accuracy_score(test_y, predictions)) # .94 - .98
