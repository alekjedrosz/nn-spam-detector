import matplotlib.pyplot as plt
import seaborn as sns


def plot_linear(x, xlabel=None, ylabel=None, line_labels=None, *args):
    """
    Plots any number of linear functions against a fixed x.
    :param x: Values for the x axis, passed as a list.
    :param xlabel: Label for the x axis.
    :param ylabel: Label for the y axis.
    :param line_labels: Labels for each of the passed in function values (in order i.e label_arg1, label_arg2, ...)
    passed as a list.
    :param args: Any number of lists of values each corresponding to one
    function to plot against x.
    """
    if not args:
        plt.plot(x)
    for arg in args:
        plt.plot(x, arg)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if line_labels is not None:
        plt.legend(line_labels)
    plt.show()


def plot_heatmap(df, title, size, one_dim=False):
    """
    Plot a heatmap given a pandas DataFrame.
    :param df: pandas DataFrame representing a heatmap.
    :param title: Title of the plot.
    :param size: Size of the figure.
    :param one_dim: set to True if your heatmap is one dimensional (e.g. correlation with target variable).
    """
    plt.figure(figsize=size)
    if one_dim:
        hm = sns.heatmap(df, xticklabels=['Target'], square=True)
        hm.set_yticklabels(df.index.values, rotation=0)
    else:
        hm = sns.heatmap(df, xticklabels=df.columns.values, yticklabels=df.columns.values, square=True)
    hm.set_title(title)
    plt.show()


def plot_cm(cm, title):
    """
    Plot a confusion matrix.
    :param cm: Confusion matrix, given as pandas DataFrame.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    hm = sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette('Blues'), square=True)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right')
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right')
    hm.set_title(title)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.show()
