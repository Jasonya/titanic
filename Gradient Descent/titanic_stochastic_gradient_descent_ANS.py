"""
File: titanic_stochastic_gradient_descent_ANS.py
Name: Jerry Liao
----------------------------------
This file contains 3 of the most important steps
in machine learning:
1) Data pre-processing
2) Training
3) Predicting
"""

import math
import time

TRAIN_DATA_PATH = 'titanic_data/train.csv'
NUM_EPOCHS = 10000
ALPHA = 0.05


def sigmoid(k):
    """
    :param k: float, linear function value
    :return: float, probability of the linear function value
    """
    return 1/(1+math.exp(-k))


def dot(lst1, lst2):
    """
    : param lst1: list, the feature vector
    : param lst2: list, the weights
    : return: float, the dot product of 2 list
    """
    total = 0
    for i in range(len(lst1)):
        total += lst1[i] * lst2[i]
    return total


def main():
    start = time.time()
    # Milestone 1
    training_data = data_pre_processing()

    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    weights = [0]*len(training_data[0][0])

    # Milestone 2
    training(training_data, weights)

    print('weights:', weights)

    # Milestone 3
    predict(training_data, weights)
    end = time.time()
    print('Total run time(Stochastic GD):', end-start)


# Milestone 1
def data_pre_processing():
    """
    Read the training data from TRAIN_DATA_PATH and get ready for training!
    :return: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    """
    all_data = []
    with open(TRAIN_DATA_PATH, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
            else:
                line = line.strip()
                feature_vector, y = feature_extractor(line)
                all_data.append((feature_vector, y))
    return all_data


def feature_extractor(line):
    """
    : param line: str, the line of data extracted from the training set
    : return: Tuple(list, label), the feature vector and the true label
    """
    data_lst = line.split(',')
    #[Id, Surv, Pclass, F_Name, L_Name, Sex5, Age, SibSp, ParCh, Ticket, Fare10, Cabin, Embarked]
    ans = []
    y = int(data_lst[1])
    for i in range(len(data_lst)):
        if i == 2:
            # Pclass
            ans.append((int(data_lst[i])-1) / (3-1))
            ans.append(((int(data_lst[i])-1) / (3-1))**2)   # degree 2
            ans.append(((int(data_lst[i]) - 1) / (3 - 1)) ** 3)  # degree 3
        elif i == 5:
            # Gender
            if data_lst[i] == 'male':
                ans.append(1)
            else:
                ans.append(0)
        elif i == 6:
            # Age
            if data_lst[i].isdigit():
                ans.append((float(data_lst[i]) - 0.42) / (80-0.42))
                ans.append(((float(data_lst[i]) - 0.42) / (80-0.42))**2)  # degree 2
                ans.append(((float(data_lst[i]) - 0.42) / (80 - 0.42)) ** 3)  # degree 3
            else:
                ans.append((30 - 0.42) / (80-0.42))
                ans.append(((30 - 0.42) / (80-0.42))**2)                 # degree 2
                ans.append(((30 - 0.42) / (80 - 0.42)) ** 3)             # degree 3
        elif i == 7:
            # SibSp
            ans.append((int(data_lst[i]) - 0) / 8)
            ans.append(((int(data_lst[i]) - 0) / 8)**2)                  # degree 2
            ans.append(((int(data_lst[i]) - 0) / 8) ** 3)                # degree 3
        elif i == 8:
            # Parch
            ans.append((int(data_lst[i]) - 0) / 6)
            ans.append(((int(data_lst[i]) - 0) / 6)**2)                  # degree 2
            ans.append(((int(data_lst[i]) - 0) / 6) ** 3)                # degree 3
        elif i == 10:
            # Fare
            ans.append((float(data_lst[i]) - 0) / 512.3292)
            ans.append(((float(data_lst[i]) - 0) / 512.3292)**2)          # degree 2
            ans.append(((float(data_lst[i]) - 0) / 512.3292) ** 3)        # degree 2
        elif i == 12:
            if data_lst[i] == 'S':
                ans.append(0 / 2)
                ans.append(0 / 2)
                ans.append(0 / 2)
            elif data_lst[i] == 'C':
                ans.append(1 / 2)
                ans.append(0.25)
                ans.append(0.125)
            elif data_lst[i] == 'Q':
                ans.append(2 / 2)
                ans.append(1)
                ans.append(1)
            else:
                ans.append(0 / 2)
                ans.append(0 / 2)
                ans.append(0)
    return ans, y


# Milestone 2
def training(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    for epoch in range(NUM_EPOCHS):
        cost = 0
        for x, y in training_data:
            #################################
            # w = w - alpha*dL_dw
            # w = w - alpha*(h-y)*x
            h = sigmoid(dot(weights, x))
            cost += -(y*math.log(h)+(1-y)*math.log(1-h))
            # w = w - alpha*(h-y)*x
            for i in range(len(weights)):
                weights[i] = weights[i] - ALPHA*(h-y)*x[i]
            #################################
        if epoch%1000 == 0:
            print('Cost over all data:', cost/len(training_data))


# Milestone 3
def predict(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    acc = 0
    num_data = 0
    for x, y in training_data:
        predict = get_prediction(x, weights)
        print('True Label: ' + str(y) + ' --------> Predict: ' + str(predict))
        if y == predict:
            acc += 1
        num_data += 1
    print('---------------------------')
    print('Acc: ' + str(acc / num_data))


def get_prediction(x, weights):
    """
    : param x: list[float], the value of each data on
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    : param weights: list[float], the weight vector (the parameters on each feature)
    : return: float, the score of x (if it is > 0 then the passenger may survive)
    """
    score = dot(x, weights)
    return 1 if score > 0 else 0


if __name__ == '__main__':
    main()
