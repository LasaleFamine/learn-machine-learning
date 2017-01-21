from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

# Simple Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Predict with test values
print(clf.predict([[160, 1]]))
