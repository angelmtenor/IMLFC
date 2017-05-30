from sklearn.linear_model import LinearRegression

size = [500, 1000, 1500]
age = [0, 20, 40]
age_size_train = [(s, a) for a in age for s in size]
price_train = [1000, 1500, 2000, 800, 1300, 1800, 600, 1100, 1600]
age_size_test = [[1500, 60]]

reg = LinearRegression()
reg.fit(age_size_train, price_train)

print("prediciton {}: {}".format(age_size_test, reg.predict(X=age_size_test)))
print("slopes: {}".format(reg.coef_))
print("intercept: {:.2f}".format(reg.intercept_))
