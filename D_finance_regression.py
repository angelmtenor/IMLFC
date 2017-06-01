"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""

# original: https://github.com/udacity/ud120-projects

import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tools.feature_format import featureFormat, targetFeatureSplit

dictionary = pickle.load(open("tools/final_project_dataset_modified.pkl", "br"))

# list the features you want to look at--first item in the
# list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)

# training-testing split needed in regression, just like classification
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

# regression
reg = LinearRegression()
reg.fit(feature_train, target_train)

print("slopes: {}".format(reg.coef_))
print("intercept: {:.2f}".format(reg.intercept_))
print("score on training data: {:.4f}".format(reg.score(feature_train, target_train)))
print("score on test data: {:.4f}".format(reg.score(feature_test, target_test)))

# draw the scatterplot, with color-coded training and testing points

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

# draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(X=feature_test))
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

# Simple regression:

# from sklearn.linear_model import LinearRegression
# 
# size = [500, 1000, 1500]
# age = [0, 20, 40]
# age_size_train = [(s, a) for a in age for s in size]
# price_train = [1000, 1500, 2000, 800, 1300, 1800, 600, 1100, 1600]
# age_size_test = [[1500, 60]]
# 
# reg = LinearRegression()
# reg.fit(age_size_train, price_train)
# 
# print("prediciton {}: {}".format(age_size_test, reg.predict(X=age_size_test)))
# print("slopes: {}".format(reg.coef_))
# print("intercept: {:.2f}".format(reg.intercept_))
