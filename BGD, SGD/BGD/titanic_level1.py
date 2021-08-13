"""
File: titanic_level1.py
Name: Jason Huang
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import util
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'
avg_list = []


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: dict[str: list], key is the column name, value is its data
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	surd, pcd, sexd, aged, sibd, parchd, fared, emd = [], [], [], [], [], [], [], []
	with open(filename, 'r') as f:
		is_first = True
		for line in f:
			if is_first:
				is_first = False
			else:
				line_list = line.strip().split(',')
				# check if mode is Train
				if mode == 'Train':
					# chek if values in Age and Embarked are Null

					if not line_list[-1] or not line_list[6]:
						continue
					else:
						surd.append(int(line_list[1]))
						pcd.append(int(line_list[2]))
						sexd.append(1 if line_list[5] == 'male' else 0)
						aged.append(float(line_list[6]))
						sibd.append(int(line_list[7]))
						parchd.append(int(line_list[8]))
						fared.append(float(line_list[10]))
						if line_list[12] == 'S':
							emd.append(0)
						elif line_list[12] == 'C':
							emd.append(1)
						elif line_list[12] == 'Q':
							emd.append(2)
				elif mode == 'Test':

					pcd.append(int(line_list[1]) if line_list[1] else avg_list[0])
					sexd.append(1 if line_list[4] == 'male' else 0)
					aged.append(float(line_list[5] if line_list[5] else avg_list[2]))
					sibd.append(int(line_list[6] if line_list[6] else avg_list[3]))
					parchd.append(int(line_list[7] if line_list[7] else avg_list[4]))
					fared.append(float(line_list[9] if line_list[9] else avg_list[5]))
					if line_list[11] == 'S':
						emd.append(0)
					elif line_list[11] == 'C':
						emd.append(1)
					elif line_list[11] == 'Q':
						emd.append(2)
					else:
						emd.append(avg_list[6])
	if mode == 'Train':
		avg_list.append(round(sum(pcd)/len(pcd), 3))
		avg_list.append(round(sum(sexd) / len(sexd), 3))
		avg_list.append(round(sum(aged) / len(sexd), 3))
		avg_list.append(round(sum(sibd)/len(sibd), 3))
		avg_list.append(round(sum(parchd)/len(parchd), 3))
		avg_list.append(round(sum(fared)/len(fared), 3))
		avg_list.append(round(sum(emd)/len(emd), 3))
		data['Survived'] = surd
	data['Pclass'] = pcd
	data['Sex'] = sexd
	data['Age'] = aged
	data['SibSp'] = sibd
	data['Parch'] = parchd
	data['Fare'] = fared
	data['Embarked'] = emd
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		sex_idx = data[feature]
		male_idx, female_idx = [], []
		for s in sex_idx:
			if s:
				male_idx.append(1)
				female_idx.append(0)
			else:
				male_idx.append(0)
				female_idx.append(1)
		data['Sex_0'] = female_idx
		data['Sex_1'] = male_idx
		data.pop(feature)
	elif feature == 'Pclass':
		pc_idx = data[feature]
		p0, p1, p2 = [], [], []
		for p in pc_idx:
			if p == 1:
				p0.append(1)
				p1.append(0)
				p2.append(0)
			elif p == 2:
				p0.append(0)
				p1.append(1)
				p2.append(0)
			elif p == 3:
				p0.append(0)
				p1.append(0)
				p2.append(1)
		data['Pclass_0'], data['Pclass_1'], data['Pclass_2'] = p0, p1, p2
		data.pop(feature)
	elif feature == 'Embarked':
		em_idx = data[feature]
		s, c, q = [], [], []
		for e in em_idx:
			if e == 0:
				s.append(1)
				c.append(0)
				q.append(0)
			elif e == 1:
				s.append(0)
				c.append(1)
				q.append(0)
			elif e == 2:
				s.append(0)
				c.append(0)
				q.append(1)
		data['Embarked_S'], data['Embarked_C'], data['Embarked_Q'] = s, c, q
		data.pop(feature)

	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for k, d in data.items():
		max_v = max(d)
		min_v = min(d)
		for i in range(len(d)):
			d[i] = (d[i] - min_v) / (max_v - min_v)
	return data


def sigmoid(k):
	return 1 / (1 + math.exp(-k))


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	Trainsamples = []
	keys = list(inputs.keys())

	def predictor(g):
		return 1 if util.dotProduct(g, weights) > 0 else 0

	if degree == 1:
		# create weight dict
		for i in range(len(keys)):
			weights[keys[i]] = 0
		# create training dict
		for i in range(len(labels)):
			attributes = {k: v[i] for k, v in inputs.items()}
			Trainsamples.append((attributes, labels[i]))

	elif degree == 2:
		# create weight dict
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
		# create training dict
		for i in range(len(labels)):
			attributes = {k: v[i] for k, v in inputs.items()}
			for j in range(len(keys)):
				for k in range(j, len(keys)):
					attributes[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]
			Trainsamples.append((attributes, labels[i]))

	# train model and predict
	for epoch in range(num_epochs):
		for x, y in Trainsamples:
			h = sigmoid(util.dotProduct(x, weights))
			util.increment(weights, -alpha * (h - y), x)
		util.evaluatePredictor(Trainsamples, predictor)

	# output weights and test result
	if degree == 1:
		f_name = 'level_1_degree1.csv'
		util.outputWeights(weights, f_name)

	elif degree == 2:
		f_name = 'level_1_degree2.csv'
		util.outputWeights(weights, f_name)
	return weights


def test_prediction(degree: int):
	"""
	This function is to create the test set prediction
	:param degree: The degree of the model
	:return:
	"""
	# read the weights file
	w_dict = defaultdict(int)
	if degree == 1:
		with open('level_1_degree1.csv', 'r') as f:
			for line in f:
				line = line.strip().split('	')
				w_dict[line[0]] = float(line[1])
	elif degree == 2:
		with open('level_1_degree2.csv', 'r') as f:
			for line in f:
				line = line.strip().split('	')
				w_dict[line[0]] = float(line[1])

	# read the preprocess the test data
	test_data = data_preprocess(TEST_FILE, {}, mode='Test')
	for f in ['Sex', 'Pclass', 'Embarked']:
		test_data = one_hot_encoding(test_data, f)
	test_data = normalize(test_data)

	attributes = {}
	Testsamples = []
	keys = list(test_data.keys())
	if degree == 1:
		# create test dict
		for i in range(len(test_data['Sex_0'])):
			attributes = {k: v[i] for k, v in test_data.items()}
			Testsamples.append(attributes)

	elif degree == 2:
		# create training dict
		for i in range(len(test_data['Sex_0'])):
			attributes = {k: v[i] for k, v in test_data.items()}
			for j in range(len(keys)):
				for k in range(j, len(keys)):
					attributes[keys[j] + keys[k]] = test_data[keys[j]][i] * test_data[keys[k]][i]
			Testsamples.append(attributes)

	def predictor(g):
		return 1 if util.dotProduct(g, w_dict) > 0 else 0

	# make prediction
	pred = [0] * len(Testsamples)
	for i in range(len(Testsamples)):
		pred[i] = predictor(Testsamples[i])

	# output the file
	if degree == 1:
		out_file(pred, 'l1_degree_1_predictions.csv')
	elif degree == 2:
		out_file(pred, 'l1_degree_2_predictions.csv')


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







