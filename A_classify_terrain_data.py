""" Classifying Terrain Data """

from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tools.class_vis import prettyPicture
from tools.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

classifiers = (GaussianNB(), SVC(kernel="rbf", C=100000.0), DecisionTreeClassifier(min_samples_split=50),
               KNeighborsClassifier(n_neighbors=10), AdaBoostClassifier(n_estimators=100), RandomForestClassifier(1000))

names = ["Naive Bayes", "SVM", "Decision Trees", "KNeighbors", "AdaBoost", "randomForest"]

for idx, clf in enumerate(classifiers):
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()

    pred = clf.predict(features_test)
    t2 = time()

    accuracy = accuracy_score(pred, labels_test)

    print("\n", clf)
    print("accuracy: \t\t {:.6f}".format(accuracy))
    print("train time(s): \t {:.6f}".format(t1 - t0))
    print("test time(s): \t {:.6f}".format(t2 - t2))

    p = prettyPicture(clf, features_test, labels_test, names[idx])

# Manual input example:

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
# clf = GaussianNB()
# clf.fit(X, Y)
# print(clf.predict([[-0.8, -1]]))
#
# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y))  # fit linear model with Stochastic Gradient Descent.
# print(clf_pf.predict([[-0.8, -1]]))
