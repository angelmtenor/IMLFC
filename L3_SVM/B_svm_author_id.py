"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.
    Use a SVC Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

# Datasets:  https://github.com/udacity/ud120-projects/tree/master/tools

import sys
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess

REDUCED = True

# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

if REDUCED:
    features_train = features_train[:len(features_train) // 100]
    labels_train = labels_train[:len(labels_train) // 100]

# your code goes here
model = SVC(kernel="rbf", C=10000, gamma='auto')
t0 = time()

model.fit(features_train, labels_train)
t1 = time()

pred = model.predict(features_test)
t2 = time()

accuracy = accuracy_score(labels_test, pred)

print("accuracy: \t\t {:.6f}".format(accuracy))
print("train time(s): \t {:.6f}".format(t1-t0))
print("test time(s): \t {:.6f}".format(t2-t2))
print("Chris emails: \t {}".format(sum(pred)))
