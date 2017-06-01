""" Classification to identify emails by their authors
    Sara has label 0
    Chris has label 1
"""

# original:  https://github.com/udacity/ud120-projects

from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tools.email_preprocess import preprocess

REDUCED = True

# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

if REDUCED:
    features_train = features_train[:len(features_train) // 10]
    labels_train = labels_train[:len(labels_train) // 10]

classifiers = (GaussianNB(), SVC(kernel="rbf", C=100000.0), DecisionTreeClassifier(min_samples_split=50),
               KNeighborsClassifier(n_neighbors=10), AdaBoostClassifier(n_estimators=100), RandomForestClassifier(1000))

names = ["Naive Bayes", "SVM", "Decision Trees", "KNeighbors", "AdaBoost", "randomForest"]

for model in classifiers:
    t0 = time()

    model.fit(features_train, labels_train)
    t1 = time()

    pred = model.predict(features_test)
    t2 = time()

    accuracy = accuracy_score(labels_test, pred)

    print("\n", model)
    print("accuracy: \t\t {:.6f}".format(accuracy))
    print("train time(s): \t {:.6f}".format(t1 - t0))
    print("test time(s): \t {:.6f}".format(t2 - t2))
    # print("Chris emails: \t {}".format(sum(pred)))
    print("number of features: {}".format(len(features_train[0])))
