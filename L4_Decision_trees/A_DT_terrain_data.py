from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time
import sys
sys.path.append("../tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = DecisionTreeClassifier()
t0 = time()

clf.fit(features_train, labels_train)
t1 = time()

pred = clf.predict(features_test)
t2 = time()

accuracy = accuracy_score(pred, labels_test)

print("accuracy: \t\t {:.6f}".format(accuracy))
print("train time(s): \t {:.6f}".format(t1-t0))
print("test time(s): \t {:.6f}".format(t2-t2))

prettyPicture(clf, features_test, labels_test)
