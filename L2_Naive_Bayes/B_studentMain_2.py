from B_classifyNB_2 import NBAccuracy
import sys
sys.path.append("../tools/")
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

print("Accuracy: {}".format(submitAccuracy()))
