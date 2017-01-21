# If you want to test with `random` choice
# import random
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class PiggyKNN():
    def fit(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train

    def predict(self, X_test):
            predictions = []
            for row in X_test:
                # Change the label choice with `random` if you want to see the difference
                # label = random.choice(self.y_train)
                label = self.closest(row)
                predictions.append(label)
            return predictions

    # Better choice than `random`
    def closest(self, row):
        # Best dist init with first point
        best_dist = euc(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        # Return the best y_train value
        return self.y_train[best_index]

# Same code of 3_iris_KN with our classifier
# Import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Split the data for train and for test from the datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Use the classifier
iris_classifier = PiggiKNN()
iris_classifier.fit(X_train, y_train)

# Test the classifier
predictions = iris_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
