"""
File: titanic_deep_nn.py
Name: Jason Huang
-----------------------------
This file demonstrates how to create a deep
neural network (5 layers NN) to train our
titanic data. Your code should use all the
constants and global variables.
You should see the following Acc if you
correctly implement the deep neural network
Acc: 0.8431372549019608
-----------------------------
X.shape = (N0, m)
Y.shape = (1, m)
W1.shape -> (N0, N1)
W2.shape -> (N1, N2)
W3.shape -> (N2, N3)
W4.shape -> (N3, N4)
W5.shape -> (N4, N5)
B1.shape -> (N1, 1)
B2.shape -> (N2, 1)
B3.shape -> (N3, 1)
B4.shape -> (N4, 1)
B5.shape -> (N5, 1)
"""

from collections import defaultdict
import numpy as np

# Constants
TRAIN = 'titanic_data/train.csv'     # This is the filename of interest
NUM_EPOCHS = 40000                   # This constant controls the total number of epochs
ALPHA = 0.01                         # This constant controls the learning rate α
L = 5                                # This number controls the number of layers in NN
NODES = {                            # This Dict[str: int] controls the number of nodes in each layer
    'N0': 6,
    'N1': 5,
    'N2': 4,
    'N3': 3,
    'N4': 2,
    'N5': 1
}


def main():
    """
    Print out the final accuracy of your deep neural network!
    You should see 0.8431372549019608
    """
    X_train, Y = data_preprocessing()
    _, m = X_train.shape
    print('Y.shape', Y.shape)
    print('X.shape', X_train.shape)
    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = normalize(X_train)
    weights, biases = neural_network(X, Y)
    K_history, A_history = {}, {'A0': X}
    for l in range(1, L + 1):
        if l == L:
            # reach the output layer
            scores = np.dot(weights['W' + str(l)].T, A_history['A' + str(l - 1)]) + biases['B' + str(l)]
            predictions = np.where(scores > 0, 1, 0)
        else:
            K_history['K' + str(l)] = np.dot(weights['W' + str(l)].T, A_history['A' + str(l - 1)]) + biases[
                'B' + str(l)]
            A_history['A' + str(l)] = np.maximum(0, K_history['K' + str(l)])
    acc = np.equal(predictions, Y)
    print('Training Acc:', np.sum(acc)/m)


def normalize(X):
    """
    :param X: numpy_array, the dimension is (num_phi, m)
    :return: numpy_array, the values are normalized, where the dimension is still (num_phi, m)
    """
    nf, m = X.shape
    for n in range(nf):
        max_val = np.max(X[n, :])
        min_val = np.min(X[n, :])
        X[n, :] = (X[n, :] - min_val) / (max_val - min_val)
    return X


def neural_network(X, Y):
    """
    :param X: numpy_array, the array holding all the training data
    :param Y: numpy_array, the array holding all the ture labels in X
    :return (weights, bias): the tuple of parameters of this deep NN
             weights: Dict[str, float], key is 'W1', 'W2', ...
                                        value is the corresponding float
             bias: Dict[str, float], key is 'B1', 'B2', ...
                                     value is the corresponding float
    """
    np.random.seed(1)
    weights = {}
    biases = {}
    n, m = X.shape
    # Initialize all the weights and biases
    for l in range(1, L+1):
        # - 0.5 to make the weights 0-centered
        weights['W' + str(l)] = np.random.rand(NODES['N' + str(l - 1)], NODES['N' + str(l)]) - 0.5
        biases['B' + str(l)] = np.random.rand(NODES['N' + str(l)], 1) - 0.5

    for epoch in range(NUM_EPOCHS):
        # Forward Pass
        K_history, A_history, cost_history, back_parameters = {}, {'A0': X}, [], {}
        for l in range(1, L+1):
            if l == L:
                # reach the output layer
                scores = np.dot(weights['W' + str(l)].T, A_history['A'+str(l-1)]) + biases['B' + str(l)]
                H = 1 / (1 + np.exp(-scores))
                lost = -(Y * np.log(H) + (1-Y) * np.log(1-H))
                J = (1/m) * np.sum(lost)
                cost_history.append(J)
            else:
                K_history['K' + str(l)] = np.dot(weights['W' + str(l)].T, A_history['A'+str(l-1)]) + biases['B' + str(l)]
                A_history['A' + str(l)] = np.maximum(0, K_history['K' + str(l)])

        if epoch % 1000 == 0:
            print('Cost', J)

        # Backward Pass
        for l in range(L, 0, -1):
            if l == L:
                # in the output layer
                back_parameters['dK' + str(l)] = (1/m) * np.sum(H-Y, axis=0, keepdims=True)
            else:
                back_parameters['dA' + str(l)] = np.dot(weights['W' + str(l+1)], back_parameters['dK' + str(l+1)])
                back_parameters['dK' + str(l)] = back_parameters['dA' + str(l)] \
                                                 * np.where(K_history['K'+str(l)] > 0, 1, 0)
            back_parameters['dW' + str(l)] = A_history['A' + str(l-1)].dot(back_parameters['dK' + str(l)].T)
            back_parameters['dB' + str(l)] = np.sum(back_parameters['dK' + str(l)], axis=1, keepdims=True)

        # Updates all the weights and biases
        for l in range(1, L+1):
            weights['W' + str(l)] = weights['W' + str(l)] - ALPHA * back_parameters['dW' + str(l)]
            biases['B'+str(l)] = biases['B'+str(l)] - ALPHA * back_parameters['dB' + str(l)]

    return weights, biases


def data_preprocessing(mode='train'):
    """
    :param mode: str, indicating if it's training mode or testing mode
    :return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
    """
    data_lst = []
    label_lst = []
    first_data = True
    if mode == 'train':
        with open(TRAIN, 'r') as f:
            for line in f:
                data = line.split(',')
                # ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
                if first_data:
                    first_data = False
                    continue
                if not data[6]:
                    continue
                label = [int(data[1])]
                if data[5] == 'male':
                    sex = 1
                else:
                    sex = 0
                # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
                passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
                data_lst.append(passenger_lst)
                label_lst.append(label)
    else:
        pass
    return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
    main()
