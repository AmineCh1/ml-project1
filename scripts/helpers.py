import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *


def partition_data(x, y, i):
    """ Partitions data according to given PRI_jet_num number.
        Args: 
            x: Data to partition.
            y: Labels to partition.
            i: PRI_jet_num number.
        Returns:
            index_i: Index of samples of data partition in original dataset. (Useful for concatenation later)
            y_i: Labels corresponding to samples with PRI_jet_num = i.
            x_i: Samples with PRI_jet_num = i. """
    #     Merge data and labels

    #     Find indexes where the 23rd element of the data is 'i' (i = 0, 1, 2 and 3)
    index_i = x[:, 22] == i

    #     Split the data according to the indexes found
    x_i = x[index_i]
    y_i = y[index_i]

    #     -999s don't appear in 1st column following the indexes, so we don't take it in consideration in the following
    #     and we get rid of the then 21st column containing the 'i's

    x_test = np.delete(x_i, 22, axis=1)

    #     Remove columns where elements have value -999
    num_cols = []

    for column in range(x_test[0].size):
        if np.all(x_test[:, column] == -999):
            num_cols.append(column)
    x_test = np.delete(x_test, num_cols, axis=1)

    #     Remove columns where all elements have the same value.(i.e single valued column)
    num_cols = []
    for column in range(x_test[0].size):
        if np.all(x_test[:, column] == x_test[0, column]):
            num_cols.append(column)
    x_test = np.delete(x_test, num_cols, axis=1)
    Print("Single valued columns:")
    print(num_cols)

    #     Concactenation of 1st column of the data and the rest with elements of value -999 removed

    #     Replace elements of value -999 in 1st column of the data by the mean of the non-NaN values of the columns

    indices = x_test[:, 0] == -999

    x_test[indices, 0] = np.mean(x_test[~indices, 0])
    print("Final shape : {} ".format(x_test.shape))
    return index_i, y_i, x_test


def compute_corr(x_test):
    """ Computes correlation matrix of given data."""
    corr_mat = np.corrcoef(x_test, rowvar=False)
    return corr_mat


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.
        Args: 
            x: """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x[:, 1:], deg)]
    return poly


def visualize_corr(corr):
    """Visualization aid for correlation matrix."""
    fig, ax = plt.subplots(figsize=(corr.shape[0], corr.shape[1]))
    sns.heatmap(corr, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()


def standardize_data(x):
    """Standardize the original data set.
        Args: 
            x: Data to standardize.
        Returns: 
            x: Standardized data.
            mean_x: Mean of the data (Sample-wise).
            std_x: Standard deviation of the data (Sample-wise).
    """

    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def run_model(y, tx, model, gamma=0.05, initial_w=[], max_iters=1000,  lambda_=0):
    """ Runs the chosen model with the appropriate parameters.

        Args:
            y:  A np array representing the labels.
            tx: A np array representing the input data.
            model: ML algorithm used. {'gd': Gradient Descent,'sgd': Stochastic Gradient Descent,
            'lq': Least Squares, 'ridge_reg': Ridge Regression,'log_reg': Logistic Regression}
            initial_w: Initial weight vector used in Gradient Descent or Stochastic Gradient Descent.
            gamma: Step size used in Gradient Descent.
            max_iters: Nb of iterations.
            lambda_: Regularizer parameter.
        Returns:
            w: Weight vector w.
    """
    # Change for future algorithms
    if model == 'gd':
        loss, w = least_squares_GD(
            y, tx, initial_w, max_iters, gamma)

    elif model == 'sgd':
        loss, w = least_squares_SGD(
            y, tx, initial_w, max_iters, gamma)

    elif model == 'lq':
        loss, w = least_squares(y, tx)

    elif model == 'ridge_reg':
        loss, w = ridge_regression(y, tx, lambda_)

    elif model == 'log_reg':
        loss, w = logistic_regression(y, tx, initial_w, max_iters, gamma)

    return w


def try_T(data, labels, prop, gamma=0.05, max_iters=100,  lambda_=0, seed=0):
    """ Separates the given data into a training set and testing set, and returns
    the accuracy of the model.

    Args:
        data: A numpy array representing the data.
        labels: A numpy array representing the labels.
        prop: Proportion of the data set dedicated to training. ( The remaining data is
        used as validation.)
        gamma: step size parameter gamma.
    Returns:
        The accuracy of the trained model on the validation set, w.r.t the chosen gamma.
    """
    N = data.shape[0]
    # We shuffle the data beforehand
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(N))
    shuffled_data = data[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]

    train_prop = int(data.shape[0]*prop)

    train = shuffled_data[:train_prop]
    y_train = shuffled_labels[:train_prop]
    y_test = shuffled_labels[train_prop:]
    test = shuffled_data[train_prop:]

#     rand_w = [np.random.uniform(-1, 1) for x in range(data.shape[1])]

    w = run_model(y_train, train, model='lq')
    return np.mean(predict_labels(w, test) == y_test)


def build_k_indices(y, k_fold, seed):
    """Builds k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(K, y, x, model,  gamma=0.05, max_iters=100,  lambda_=0, logging=True, seed=0.8):
    """ Applies cross validation to a given set of data.

        Args:
            K: Number of folds.
            y_batch: Labels of given dataset.
            tx_batch: Dataset.
            max_iters: Number of iterations run for each fold.
            gamma: step size parameter gamma.
            model:  ML algorithm used. {'gd': Gradient Descent,'sgd': Stochastic Gradient Descent,
            'lq': Least Squares, 'ridge_reg': Ridge Regression,'log_reg': Logistic Regression}
            initial_w: Initial weight vector used in Gradient Descent or Stochastic Gradient Descent.
            logging: Intermediary printing.
        Returns:
            A mean of the accuracies of the K different folds.
    """
    k_indices = build_k_indices(y, K, seed)
    accuracies = []  # nb of accuracies to compute = K
    ws = []

    for k in range(K):
        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)

        y_te = y[te_indice]
        y_tr = y[tr_indice]
        x_te = x[te_indice]
        x_tr = x[tr_indice]

        w = run_model(y_tr, x_tr, model=model, lambda_=lambda_)
        acc = np.mean(predict_labels(w, x_te) == y_te)
        ws.append(w)
        accuracies.append(acc)

        if logging:
            print(acc)
    if logging:
        print("--------------")
    return np.mean(accuracies)


def forward_selection(y, tx, K):
    """ Implements forward selection of the 2nd order, that is, taking the product of two columns and appending it to tx, and 
        assessing whether the resulting matrix is better for training our chosen model.
        Args: 
            y: Labels to predict.
            tx: Data to iteratively augment with interaction terms.
            K: Number of folds used in the cross-validation.
        Returns:
            tx: Possibly augmented tx.
        """"
    n_col = tx.shape[1]
    indices = [[i, j] for i in range(n_col) for j in range(n_col) if j >= i]

    basis_acc = cross_validation(K, y, tx, 'lq')
    for idx in indices:

        augmented_tx = np.c_[tx, tx[:, idx[0]] * tx[:, idx[1]]]
        augmented_acc = (cross_validation(K, augmented_tx, y, 'lq'))

        if augmented_acc > basis_acc:
            basis_acc = augmented_acc
            tx = augmented_tx

    return tx
