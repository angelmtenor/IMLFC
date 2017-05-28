def classify(features_train, labels_train):

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()  # create classifier
    clf.fit(features_train, labels_train)  # fit the classifier on the training features and labels
    return clf  # return the fit classifier
