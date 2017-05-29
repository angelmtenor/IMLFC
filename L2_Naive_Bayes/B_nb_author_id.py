"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

# Datasets:  https://github.com/udacity/ud120-projects/tree/master/tools

import sys
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# your code goes here
model = GaussianNB()
t0 = time()

model.fit(features_train, labels_train)
t1 = time()

pred = model.predict(features_test)
t2 = time()

accuracy = accuracy_score(labels_test, pred)

print("accuracy: \t\t {:.6f}".format(accuracy))
print("train time(s): \t {:.6f}".format(t1-t0))
print("test time(s): \t {:.6f}".format(t2-t2))
