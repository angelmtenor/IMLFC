from prep_terrain_data import makeTerrainData
from classifyNB_2 import NBAccuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

print("Accuracy: {}".format(submitAccuracy()))