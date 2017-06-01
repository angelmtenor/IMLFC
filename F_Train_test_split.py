"""
PLEASE NOTE:
The api of train_test_split changed and moved from sklearn.cross_validation to
sklearn.model_selection(version update from 0.17 to 0.18)
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from sklearn import cross_validation  # for version 0.17

iris = datasets.load_iris()
features = iris.data
labels = iris.target

# import the relevant code and make your train/test split
# name the output datasets features_train, features_test,
# labels_train, and labels_test

# set the random_state to 0 and the test_size to 0.4
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=0)

clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)
print("score: {:.4f}".format(clf.score(features_test, labels_test)))
