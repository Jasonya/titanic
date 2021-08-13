"""
File: boston_housing_competition.py
Name: Jason Huang
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!
"""
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
TRAIN = 'boston_housing/train.csv'
TEST = 'boston_housing/test.csv'


def main():
	# read the data
	data = pd.read_csv(TRAIN)

	print('Check values in columns:')
	print(data.isnull().any())
	print(data.corr())

	# choose lstat, rm, ptratio, indus, which has higher corr with labels
	selected_attributes = ['lstat', 'rm', 'ptratio', 'indus']
	y = data['medv']
	x = data[selected_attributes]
	# x = data.drop(['medv'],axis=1)


	# do the StandardScaler
	scaler = preprocessing.StandardScaler()
	x = scaler.fit_transform(x)

	# split the train data set to training and validation sets
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=20, shuffle=True)

	# model 1 Linear Regression
	model1 = make_pipeline(preprocessing.PolynomialFeatures(degree=3), linear_model.LinearRegression())
	model1.fit(x_train, y_train)

	print(' model 1 training set RMSE:{}'.format(mean_squared_error(model1.predict(x_train), y_train)**0.5))
	print(' model 1 validation set RMSE:{}'.format(mean_squared_error(model1.predict(x_test), y_test)**0.5))

	# model 2 Support Vector Machine
	model2 = SVR(kernel='rbf', C=1e3, gamma=0.1)
	model2.fit(x_train, y_train)

	print(' model 2 training set RMSE:{}'.format(mean_squared_error(model2.predict(x_train), y_train) ** 0.5))
	print(' model 2 validation set RMSE:{}'.format(mean_squared_error(model2.predict(x_test), y_test) ** 0.5))

	# model 3 Decision Tree Regression
	model3 = DecisionTreeRegressor(max_depth=3)
	model3.fit(x_train, y_train)

	print(' model 3 training set RMSE:{}'.format(mean_squared_error(model3.predict(x_train), y_train) ** 0.5))
	print(' model 3 validation set RMSE:{}'.format(mean_squared_error(model3.predict(x_test), y_test) ** 0.5))

	# model 4 KNN Regression
	model4 = KNeighborsRegressor(n_neighbors=3)
	model4.fit(x_train, y_train)

	print(' model 4 training set RMSE:{}'.format(mean_squared_error(model4.predict(x_train), y_train) ** 0.5))
	print(' model 4 validation set RMSE:{}'.format(mean_squared_error(model4.predict(x_test), y_test) ** 0.5))

	# model 5 Random Forest Regression
	model5 = RandomForestRegressor(n_estimators=30)
	model5.fit(x_train, y_train)

	print(' model 5 training set RMSE:{}'.format(mean_squared_error(model5.predict(x_train), y_train) ** 0.5))
	print(' model 5 validation set RMSE:{}'.format(mean_squared_error(model5.predict(x_test), y_test) ** 0.5))

	# Gradient Boosting Regression
	model6 = GradientBoostingRegressor(n_estimators=32)
	model6.fit(x_train, y_train)
	print(' model 6 training set RMSE:{}'.format(mean_squared_error(model6.predict(x_train), y_train) ** 0.5))
	print(' model 6 validation set RMSE:{}'.format(mean_squared_error(model6.predict(x_test), y_test) ** 0.5))

	# predict the test set
	test_data = pd.read_csv(TEST)

	# record the ID and do the StandardScaler
	out = test_data['ID']
	tx = test_data[selected_attributes]
	tx = scaler.transform(tx)

	# make the predictions by each model
	pred_1 = model1.predict(tx)
	pred_2 = model2.predict(tx)
	pred_3 = model3.predict(tx)
	pred_4 = model4.predict(tx)
	pred_5 = model5.predict(tx)
	pred_6 = model6.predict(tx)

	# combine the ID with the predicted values, then output
	out1 = pd.concat([out, pd.DataFrame(pred_1, columns=['medv'])], axis=1)
	out2 = pd.concat([out, pd.DataFrame(pred_2, columns=['medv'])], axis=1)
	out3 = pd.concat([out, pd.DataFrame(pred_3, columns=['medv'])], axis=1)
	out4 = pd.concat([out, pd.DataFrame(pred_4, columns=['medv'])], axis=1)
	out5 = pd.concat([out, pd.DataFrame(pred_5, columns=['medv'])], axis=1)
	out6 = pd.concat([out, pd.DataFrame(pred_6, columns=['medv'])], axis=1)
	out1.to_csv('model1_predictions.csv', index=False)
	out2.to_csv('model2_predictions.csv', index=False)
	out3.to_csv('model3_predictions.csv', index=False)
	out4.to_csv('model4_predictions.csv', index=False)
	out5.to_csv('model5_predictions.csv', index=False)
	out6.to_csv('model6_predictions.csv', index=False)


if __name__ == '__main__':
	main()
