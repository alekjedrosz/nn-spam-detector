import pandas as pd
import numpy as np

import plotting_tools
import preprocessing
from neural_net import NeuralNetwork


def run():
    df = pd.read_csv('spambase_data\\spambase.data', header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    print(f'No. missing values: {df.isnull().sum().sum()}')

    X = df.drop(57, axis=1)
    y = df.loc[:, 57]

    # Feature selection

    abs_corr_w_target = X.apply(lambda col: col.corr(y)).abs().sort_values().to_frame()
    abs_corr = X.corr().abs()
    plotting_tools.plot_heatmap(abs_corr_w_target, title='Correlation with target', size=(8, 16), one_dim=True)
    plotting_tools.plot_heatmap(abs_corr, title='Correlation before feature selection', size=(10, 16))

    to_drop = set()

    # Amount of variation
    variance = X.var(axis=0, ddof=1)
    to_drop.update(variance[variance < 0.01].index.values)

    # Correlation with target
    to_drop.update(abs_corr_w_target[abs_corr_w_target[0] < 0.01].index.values)

    # Pairwise correlation
    to_drop.update(preprocessing.find_correlated(abs_corr, abs_corr_w_target))

    to_drop = list(to_drop)
    nr_dropped = len(to_drop)
    X.drop(to_drop, axis=1, inplace=True)
    abs_corr = X.corr().abs()
    plotting_tools.plot_heatmap(abs_corr, title='Correlation after feature selection', size=(10, 16))
    print(f'Dropped features: {to_drop}')

    # Data standardization works better, use normalization only for tests
    # X = preprocessing.normalize_data(X)
    X = preprocessing.standardize_data(X)

    X = X.values
    y = y.values
    train_inputs, cv_inputs, test_inputs = np.split(X, [int(0.6 * len(df)), int(0.8 * len(df))])
    train_outputs, cv_outputs, test_outputs = np.split(y, [int(0.6 * len(df)), int(0.8 * len(df))])

    print(f'Training set size: {train_outputs.shape[0]}\n'
          f'Cross validation set size: {cv_outputs.shape[0]}\n'
          f'Test set size: {test_outputs.shape[0]}')

    model = NeuralNetwork([57 - nr_dropped, 32, 1], activation_function='sigmoid')

    # Only use this part for tuning hyperparameters, slows down the program significantly
    # lambdas = list(np.arange(0.5, 1.5, 0.1))
    # model.plot_learning_curves(train_inputs, train_outputs, cv_inputs, cv_outputs,
    #                           learning_rate=1.5, epochs=500, lambda_=0.6)
    # model.plot_validation_curves(train_inputs, train_outputs, cv_inputs, cv_outputs,
    #                             learning_rate=1.5, epochs=1000, lambdas=lambdas)

    model.gradient_descent(train_inputs, train_outputs, 1.5, 4000, 0.6, gradient_check=False, plot_cost=False)

    train_predictions = np.where(model.predict(train_inputs) > 0.5, 1, 0)
    test_predictions = np.where(model.predict(test_inputs) > 0.5, 1, 0)

    train_columns = {'Train predictions': train_predictions[:, 0], 'Train outputs': train_outputs}
    test_columns = {'Test predictions': test_predictions[:, 0], 'Test outputs': test_outputs}

    train_results = pd.DataFrame(train_columns)
    test_results = pd.DataFrame(test_columns)

    train_correct = pd.value_counts(train_results['Train predictions'] == train_results['Train outputs'])[True]
    test_correct = pd.value_counts(test_results['Test predictions'] == test_results['Test outputs'])[True]

    test_positive_predictions = test_results[test_results['Test predictions'] == 1]
    test_negative_predictions = test_results[test_results['Test predictions'] == 0]

    test_is_positive_correct = pd.value_counts(test_positive_predictions['Test predictions']
                                               == test_positive_predictions['Test outputs'])
    test_is_negative_correct = pd.value_counts(test_negative_predictions['Test predictions']
                                               == test_negative_predictions['Test outputs'])

    test_true_positives = test_is_positive_correct[True]
    test_false_positives = test_is_positive_correct[False]
    test_true_negatives = test_is_negative_correct[True]
    test_false_negatives = test_is_negative_correct[False]

    test_precision = test_true_positives / (test_true_positives + test_false_positives)
    test_recall = test_true_positives / (test_true_positives + test_false_negatives)
    test_confusion_matrix = pd.DataFrame([[test_true_positives, test_false_positives],
                                          [test_false_negatives, test_true_negatives]],
                                         columns=[1, 0],
                                         index=[1, 0])

    train_acc = train_correct / len(train_outputs)
    test_acc = test_correct / len(test_outputs)
    print(f'train_acc = {train_acc}')
    print(f'test_acc = {test_acc}')
    print(f'test_precision = {test_precision}')
    print(f'test_recall = {test_recall}')

    plotting_tools.plot_cm(test_confusion_matrix, title='Confusion matrix')


if __name__ == '__main__':
    run()
