"""
File: titanic_level2.py
Name: Jason Huang
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'
fillna_ref = {}


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	interest = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	if mode == 'Train':
		data = data[interest].dropna(axis=0)
		labels = data[interest[0]]
	elif mode == 'Test':
		data = data[interest[1:]]

	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2
	if mode == 'Train':
		for i in interest:
			if data[i].mean().is_integer():
				fillna_ref[i] = int(data[i].mean())
			else:
				fillna_ref[i] = round(data[i].mean(), 3)

		return data[interest[1:]], labels
	elif mode == 'Test':
		data.fillna(value=fillna_ref, inplace=True)
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		data['Sex_0'], data['Sex_1'] = 0, 0
		data.loc[data.Sex == 1, 'Sex_1'] = 1
		data.loc[data.Sex == 0, 'Sex_0'] = 1
		data.drop([feature], axis=1)
	elif feature == 'Pclass':
		data['Pclass_0'], data['Pclass_1'], data['Pclass_2'] = 0, 0, 0
		data.loc[data.Pclass == 1, 'Pclass_0'] = 1
		data.loc[data.Pclass == 2, 'Pclass_1'] = 1
		data.loc[data.Pclass == 3, 'Pclass_2'] = 1
		data.drop([feature], axis=1)
	elif feature == 'Embarked':
		data['Embarked_0'], data['Embarked_1'], data['Embarked_2'] = 0, 0, 0
		data.loc[data.Embarked == 'S', 'Embarked_0'] = 1
		data.loc[data.Embarked == 'C', 'Embarked_1'] = 1
		data.loc[data.Embarked == 'Q', 'Embarked_2'] = 1
		data.drop([feature], axis=1)
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	scaler = preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> 0.8019662921348315
	TODO: real accuracy on degree2 -> 0.8314606741573034
	TODO: real accuracy on degree3 -> 0.8792134831460674
	"""
	for d in ['degree_1', 'degree_2', 'degree_3']:
		x_train, y_train = data_preprocess(TRAIN_FILE)
		x_test = data_preprocess(TEST_FILE, mode='Test')
		for f in ['Sex', 'Pclass', 'Embarked']:
			x_train = one_hot_encoding(x_train, f)
			x_test = one_hot_encoding(x_test, f)
		scaler = preprocessing.StandardScaler()
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.transform(x_test)

		# degree 1
		if d == 'degree_1':
			model = linear_model.LogisticRegression(max_iter=10000).fit(x_train, y_train)
			print('Degree 1 Accuracy :{}'.format(model.score(x_train, y_train)))
			pred = model.predict(x_test)
			out_file(pred, 'degree_1_predictions.csv')

		# degree 2
		if d == 'degree_2':
			poly = preprocessing.PolynomialFeatures(degree=2)
			x_train = poly.fit_transform(x_train)
			x_test = poly.transform(x_test)
			model = linear_model.LogisticRegression(max_iter=10000).fit(x_train, y_train)
			print('Degree 2 Accuracy :{}'.format(model.score(x_train, y_train)))
			pred = model.predict(x_test)
			out_file(pred, 'degree_2_predictions.csv')

		# degree 3
		if d == 'degree_3':
			poly = preprocessing.PolynomialFeatures(degree=3)
			x_train = poly.fit_transform(x_train)
			x_test = poly.transform(x_test)
			model = linear_model.LogisticRegression(max_iter=10000).fit(x_train, y_train)
			print('Degree 3 Accuracy :{}'.format(model.score(x_train, y_train)))
			pred = model.predict(x_test)
			out_file(pred, 'degree_3_predictions.csv')


def out_file(predictions, filename):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('PassengerId,Survived\n')
		start_id = 892
		for ans in predictions:
			out.write(str(start_id)+','+str(ans)+'\n')
			start_id += 1
	print('===============================================')

if __name__ == '__main__':
	main()
