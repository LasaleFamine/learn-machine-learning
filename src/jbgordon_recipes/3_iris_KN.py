# Import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Split the data for train and for test from the datasets using the train_test_split method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Use the simple DecisionTreeClassifier
# from sklearn import tree
# iris_classifier = tree.DecisionTreeClassifier()

# Or use another simple classifier
from sklearn.neighbors import KNeighborsClassifier
iris_classifier = KNeighborsClassifier()

iris_classifier.fit(X_train, y_train)

# Test with the classifier
predictions = iris_classifier.predict(X_test)

# Print accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
