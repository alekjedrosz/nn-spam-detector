import numpy as np

import plotting_tools


class NeuralNetwork:
    def __init__(self, shape, activation_function='sigmoid'):
        self.shape = shape
        self.num_layers = len(shape)
        self.weights = None
        self.initialize_weights()
        self.activation_fn = globals()[activation_function]
        self.activation_del = globals()[''.join([activation_function, '_derivative'])]

    def initialize_weights(self):
        """
        Randomly initializes weights to values in range [-epsilon, epsilon],
        calculating epsilon based on the number of neurons in the layers this
        weight matrix operates on.
        """
        self.weights = []
        for s_in, s_out in zip(self.shape[:-1], self.shape[1:]):
            epsilon = np.sqrt(6) / (np.sqrt(s_in + s_out))
            self.weights.append(np.random.rand(s_out, s_in + 1) * (2 * epsilon) - epsilon)

    def feedforward(self, X, weights=None, predict=False):
        """
        Computes the output of the network, given a matrix of input vectors. Fully vectorized
        implementation, compatible with entire matrices of training examples.
        :param X: Matrix of input vectors (one training example per row).
        :param weights: Optional, custom weights to use for the computation. If not given
        uses the network's global weights.
        :param predict: If True, returns only the activations in the output layer.
        :return: Activations and weighted sums of all layers in the network (unless 'predict' is set
        to True).
        """
        if weights is None:
            weights = self.weights
        A = np.c_[np.ones((np.shape(X)[0], 1)), X]
        A = A.T
        activations = [A]
        weighted_sums = []
        for layer, W in enumerate(weights, 1):
            Z = W.dot(A)
            if layer < len(weights):
                A = self.activation_fn(Z)
                A = np.r_[np.ones((1, np.shape(A)[1])), A]
            else:
                A = sigmoid(Z)
            activations.append(A)
            weighted_sums.append(Z)
        if predict:
            return activations[-1].T
        else:
            return activations, weighted_sums

    def backpropagate(self, X, Y, lambda_=0):
        """
        Efficient backpropagation implementation, computes derivatives of the
        cost function wrt. every weight (parameter) in the network. Also compatible
        with entire matrices of input / output vectors and fully vectorized.
        :param X: Matrix of input vectors (one training example per row).
        :param Y: Matrix of expected output vectors (one training example per row).
        :param lambda_: Regularization parameter.
        :return: Gradient of the cost function (derivatives wrt. every parameter), as a
        list of matrices with shapes corresponding to those of the weight matrices
        in the network.
        """
        m = len(X)
        activations, weighted_sums = self.feedforward(X)
        E_curr = cross_entropy_derivative(Y, activations.pop())
        gradient = []

        gradient.append(E_curr.dot(activations.pop().T) / m)
        weighted_sums.pop()

        for W in reversed(self.weights[1:]):
            E_next = W[:, 1:].T.dot(E_curr) * self.activation_del(weighted_sums.pop())
            E_curr = E_next
            gradient.append(E_curr.dot(activations.pop().T) / m)
        gradient = list(reversed(gradient))
        regularization_terms = [(lambda_ / m) * np.c_[np.zeros((len(W), 1)), W[:, 1:]]
                                for W in self.weights]
        gradient = [G + R for G, R in zip(gradient, regularization_terms)]
        return gradient

    def gradient_descent(self, X, Y, learning_rate, epochs, lambda_=0,
                         gradient_check=False, plot_cost=False):
        """
        Performs a minimization of the cross entropy cost over the network's weights using
        batch gradient descent. Uses fully vectorized forward- and back-propagation for
        computational efficiency. Weights are randomly initialized with each call to the function.
        :param X: Matrix of input vectors (typically from a training set).
        :param Y: Matrix of expected output vectors (typically from a training set).
        :param learning_rate: Learning rate for minimization.
        :param epochs: Number of epochs of learning.
        :param lambda_: Regularization parameter (default=0).
        :param gradient_check: If True computes numerical (approximated) gradient and
        compares it against analytical (backpropagated) gradient. Useful for testing
        backpropagation correctness.
        :param plot_cost: If True plots total cost of the network against the number of epochs.
        """
        self.initialize_weights()
        if plot_cost:
            costs = []
        if gradient_check:
            self.gradient_check(X, Y, lambda_)

        for epoch in range(epochs):
            gradient = self.backpropagate(X, Y, lambda_)
            self.weights = [W - learning_rate * G for W, G in zip(self.weights, gradient)]
            if plot_cost:
                costs.append(self.cost(X, Y, lambda_))
        if plot_cost:
            plotting_tools.plot_linear(list(range(1, epochs + 1)), 'Epoch', 'Cost', ['Cost'], costs)

    def cost(self, X, Y, lambda_=0, weights=None, regularize=False):
        """
        Compute the cross entropy cost of the network, optionally regularized with L2 regularization.
        :param X: Matrix of input vectors.
        :param Y: Matrix of expected output vectors.
        :param lambda_: Regularization parameter.
        :param weights: Custom weights to use for the computation, if not provided the
        network's weights are used.
        :param regularize: If true computes the regularized cost using L2 regularization.
        :return: The total cost of the network.
        """
        if weights is None:
            weights = self.weights
        m = len(X)
        activations, _ = self.feedforward(X, weights)
        output_activations = activations[-1]
        if regularize:
            weights_no_bias = [W[:, 1:] for W in weights]
            regularization_sum = sum([np.sum((lambda_ / (2 * m)) * (W * W)) for W in weights_no_bias])
            cost = cross_entropy_cost(Y, output_activations) + regularization_sum
        else:
            cost = cross_entropy_cost(Y, output_activations)
        return cost

    def predict(self, X):
        """
        Predict the output for each input vector, passed in as a matrix of vectors.
        :param X: Matrix of input vectors (one vector per row).
        :return: Matrix of outputs for every input vector (one output vector per row).
        """
        return self.feedforward(X, predict=True)

    def gradient_check(self, X, Y, lambda_=0):
        """
        Compute and print out the numerical gradient, the analytical gradient and
        the relative difference between them. For a correct backpropagation implementation
        the relative difference should be small (depending on the number of examples).
        :param X: Input matrix.
        :param Y: Expected output vectors.
        :param lambda_: Regularization parameter.
        """
        analytical_gradient = self.backpropagate(X, Y, lambda_)
        numerical_gradient = self.numerical_gradient(X, Y, lambda_)
        analytical_gradient = np.concatenate([G.ravel() for G in analytical_gradient])
        numerical_gradient = np.concatenate([G.ravel() for G in numerical_gradient])
        relative_difference = np.linalg.norm((numerical_gradient - analytical_gradient)) \
                              / np.linalg.norm((numerical_gradient + analytical_gradient))
        print(f'\nAnalytical gradient:\n{analytical_gradient}\n')
        print(f'Numerical gradient:\n{numerical_gradient}\n')
        print(f'Relative difference:\n{relative_difference}\n')

    def numerical_gradient(self, X, Y, lambda_=0):
        """
        Naive implementation of numerical (approximated) gradient for gradient checking.
        :param X: Input matrix
        :param Y: Expected output vectors
        :param lambda_: Regularization parameter
        :return: List of numerical gradients wrt. every parameter.
        """
        params = [W.ravel(order='F') for W in self.weights]
        numgrad = [np.zeros(np.shape(W)) for W in params]
        epsilon = 0.0001
        for i in range(len(params)):
            for j in range(len(params[i])):
                params[i][j] += epsilon
                reshaped_params = [P.reshape(np.shape(W), order='F') for P, W in zip(params, self.weights)]
                loss1 = self.cost(X, Y, lambda_, weights=reshaped_params, regularize=True)
                params[i][j] -= epsilon
                params[i][j] -= epsilon
                reshaped_params = [P.reshape(np.shape(W), order='F') for P, W in zip(params, self.weights)]
                loss2 = self.cost(X, Y, lambda_, weights=reshaped_params, regularize=True)
                params[i][j] += epsilon
                numgrad[i][j] = (loss1 - loss2) / (2 * epsilon)
        return [G.reshape(np.shape(W), order='F') for G, W in zip(numgrad, self.weights)]

    def plot_learning_curves(self, X_train, Y_train, X_cv, Y_cv, learning_rate, epochs, lambda_):
        """
        Plots training set error and cross validation set error against the number of examples
        the model has been trained with. Artificially limits the training set to a subset of
        training examples in each iteration (increasing it over time), trains a model using that
        subset and calculates the cost on both, the training set and the cross validation set.
        Errors aggregated in such way are then plotted against the number of training examples
        used in training that model.
        :param X_train: Training set inputs.
        :param Y_train: Training set outputs.
        :param X_cv: Cross validation set inputs.
        :param Y_cv: Cross validation set outputs.
        :param learning_rate: Learning rate for minimization.
        :param epochs: Epochs of learning for minimization.
        :param lambda_: Regularization parameter.
        """
        m = len(X_train)
        m = m if m < 1000 else 1000
        training_errors = []
        cv_errors = []
        for i in range(1, m + 1, 5):
            X_train_subset = X_train[:i, :]
            Y_train_subset = Y_train[:i]
            self.gradient_descent(X_train_subset, Y_train_subset, learning_rate, epochs, lambda_)
            training_cost = self.cost(X_train_subset, Y_train_subset)
            cv_cost = self.cost(X_cv, Y_cv)
            training_errors.append(training_cost)
            cv_errors.append(cv_cost)
        plotting_tools.plot_linear(list(range(1, m + 1, 5)), 'Training examples', 'Cost',
                                   ['Training error', 'CV error'], training_errors, cv_errors)

    def plot_validation_curves(self, X_train, Y_train, X_cv, Y_cv, learning_rate, epochs, lambdas):
        """
        Plots training set error and cross validation set error against the given values of lambda,
        sorted in ascending order.
        :param X_train: Training set inputs.
        :param Y_train: Training set outputs.
        :param X_cv: Cross validation set inputs.
        :param Y_cv: Cross validation set outputs.
        :param learning_rate: Learning rate for minimization.
        :param epochs: Epochs of learning for minimization.
        :param lambdas: Lambda values to try.
        """
        lambdas.sort()
        training_errors = []
        cv_errors = []
        for lambda_ in lambdas:
            training_cost = 0
            cv_cost = 0
            for i in range(10):
                print(lambda_)
                self.gradient_descent(X_train, Y_train, learning_rate, epochs, lambda_)
                training_cost += self.cost(X_train, Y_train, lambda_)
                cv_cost += self.cost(X_cv, Y_cv, lambda_)
            training_errors.append(training_cost / 10)
            cv_errors.append(cv_cost / 10)
        plotting_tools.plot_linear(lambdas, 'Lambda', 'Cost', ['Training error', 'CV error'],
                                   training_errors, cv_errors)


def cross_entropy_cost(Y, A):
    m = len(Y)
    Y = Y.T
    return (-1 / m) * np.sum(np.nan_to_num(Y * np.log(A) + (1 - Y) * np.log(1 - A)))


def cross_entropy_derivative(Y, A):
    """
    Derivative with respect to the activations in the output layer.
    :param Y: Expected output activations.
    :param A: Output activation
    :return: Derivatives for all training examples.
    """
    return A - Y.T


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_derivative(Z):
    g = sigmoid(Z)
    return g * (1 - g)


def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def tanh_derivative(Z):
    g = tanh(Z)
    return 1 - (g ** 2)


def relu(Z):
    return Z * (Z > 0)


def relu_derivative(Z):
    return 1 * (Z > 0)
