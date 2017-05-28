def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ Compute the accuracy of your Naive Bayes classifier """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    clf = GaussianNB()  # create classifier
    clf.fit(features_train, labels_train)  # fit the classifier on the training features and labels
    pred = clf.predict(features_test)  # use the trained classifier to predict labels for the test features
    accuracy = accuracy_score(labels_test, pred)  # calculate and return the accuracy on the test data
    # Other way: accuracy = clf.score(features_test, labels_test)

    return accuracy
