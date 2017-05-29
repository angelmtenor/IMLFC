"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

# Datasets:  https://github.com/udacity/ud120-projects/tree/master/tools

import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# your code goes here
model = GaussianNB()
model.fit(features_train, labels_train)
pred = model.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print("accuracy: {:.3f}".format(accuracy))
