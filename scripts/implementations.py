import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from proj1_helpers import *


def sigmoid(x):
    """Applies sigmoid function to x."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss_MSE(y, tx, w):
    """Computes the mean squared error."""
    e = y - tx.dot(w)
    return 1/2 * np.mean(e**2)


def compute_loss_MAE(y, tx, w):
    """Computes the mean absolute error."""
    e = y - tx.dot(w)
    return np.mean(np.abs(e))


def compute_loss_rmse(y, tx, w):
    """Computes the root mean squared error."""
    return np.sqrt(2 * compute_loss_MSE(y, tx, w))


def compute_loss_logistic(y, tx, w):
    """Computes the loss by negative log likelihood."""
    prediction = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(prediction)) + (1 - y).T.dot(np.log(1 - prediction))
    return np.squeeze(-loss)


def compute_gradient(y, tx, w):
    """Computes the gradient of the weight vector for linear regression."""
    N = len(y)
    e = y - tx @ w
    return (-1 / N) * tx.T @ e


def compute_gradient_logistic(y, tx, w):
    """Computes the gradient of the logistic regression loss."""
    prediction = sigmoid(tx.dot(w))
    grad = tx.T.dot(prediction - y)
    return grad


def compute_subgradient(y, tx, w):
    # In case we want to play with non-differentiable functions
    N = len(y)
    e = y - tx@w
    return (-1/N)*np.sign(e)@tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the
    input data 'tx').
    Outputs an iterator which gives mini-batches of `batch_size` matching
    elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data
    messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_GD(y, tx, initial_w, max_iters, gamma, logging=False):
    """Computes a least squares model using gradient descent.

    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
        initial_w: A numpy array representing the initial weights of each
            feature.
        max_iters: An integer specifying the maximum number of iterations for
            convergence.
        gamma: A float number greater than 0 used as a learning rate.
        logging: A boolean to print logs or not.

    Returns:
        A tuple with last weight and loss respectively.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter_ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss_MSE(y, tx, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        if logging:
            print(
                f"Gradient Descent({iter_}/{max_iters - 1}): loss={loss}, w={w}")
    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, logging=False):
    """Computes a least squares model using stochastic gradient descent.
    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
        initial_w: A numpy array representing the initial weights of each
            feature.
        max_iters: An integer specifying the maximum number of iterations for
            convergence.
        gamma: A float number greater than 0 used as a learning rate.
        logging: A boolean to print logs or not.

    Returns:
        A tuple with last weight and loss respectively.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter_ in range(max_iters):
        # This for loop has one iteration.
        for min_batch_y, min_batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_gradient(min_batch_y, min_batch_tx, w)
            loss = compute_loss_MSE(y, tx, w)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
            if logging == True:
                print(f"SGD({iter_}/{max_iters - 1}): loss={loss}, w={w}")
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """Calculates the least squares using the normal equations.
    
    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
    Returns:
        A tuple with last weight and loss respectively.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Computes rigde regression using normal equations.

    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
        lambda_: A float representing the tradeoff parameter for L2-norm
            regularization.
    Returns:
        A tuple with last weight and loss respectively.
    """
    lambda_prime = 2 * lambda_ * len(y)
    a = tx.T @ tx + (lambda_prime * np.identity(tx.shape[1]))
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss_rmse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, logging=False):
    """Computes logistic regression using gradient descent.
    
    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
        initial_w: A numpy array representing the initial weights of each
            feature.
        max_iters: An integer specifying the maximum number of iterations for
            convergence.
        gamma: A float number greater than 0 used as a learning rate.
        logging: A boolean to print logs or not.
    Returns:
        A tuple with last weight and loss respectively.
    """
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
    if len(initial_w.shape) == 1:
        initial_w = np.expand_dims(initial_w, axis=1)
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        loss = compute_loss_logistic(y, tx, w)
        if loss == np.inf:
            if logging:
                print(f'Stopped at {i} with previous loss {losses[-2]}')
            return ws[-2], losses[-2]
        if i % 10000 == 0:
            if logging:
                print(f"At {i} with loss={loss}")
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
    return ws[-1], losses[-1]


def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the loss, gradient"""
    num_samples = y.shape[0]
    loss = compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, logging=False):
    """Computes L2-norm regularized logistic regression using gradient descent.
    
    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
        lambda_: A float representing the tradeoff parameter for L2-norm
            regularization.
        initial_w: A numpy array representing the initial weights of each
            feature.
        max_iters: An integer specifying the maximum number of iterations for
            convergence.
        gamma: A float number greater than 0 used as a learning rate.
        logging: A boolean to print logs or not.
    Returns:
        A tuple with last weight and loss respectively.
    """
    # Transform column vectors with shape of the form (M,) to (M, 1)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
    if len(initial_w.shape) == 1:
        initial_w = np.expand_dims(initial_w, axis=1)
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if logging:
            if i % 1000 == 0:
                print(f'At step {i} with loss={loss}')
        losses.append(loss)
        ws.append(w)
    return ws[-1], losses[-1] 


############################ ADDITIONAL FUNCTIONS #############################


def partition_data(x, y, i):
    """Partitions data according to given PRI_jet_num number.

    Args: 
        x: Data to partition.
        y: Labels to partition.
        i: PRI_jet_num number.

    Returns:
        index_i: Index of samples of data partition in original dataset.
            (Useful for concatenation later)
        y_i: Labels corresponding to samples with PRI_jet_num = i.
        x_i: Samples with PRI_jet_num = i.
    """
    # Find indexes where the 23rd element of the data is 'i' (i = 0, 1, 2 or 3)
    index_i = x[:, 22] == i
    # Split the data according to the indexes found
    x_i = x[index_i]
    y_i = y[index_i]
    # -999s don't appear in 1st column following the indexes, so we don't take
    # it in consideration in the following and we get rid of the then 21st
    # column containing the 'i's
    x_test = np.delete(x_i, 22, axis=1)
    # Remove columns where elements have value -999
    num_cols = []
    for column in range(x_test[0].size):
        if np.all(x_test[:, column] == -999):
            num_cols.append(column)
    x_test = np.delete(x_test, num_cols, axis=1)
    # Remove single valued columns
    num_cols = []
    for column in range(x_test[0].size):
        if np.all(x_test[:, column] == x_test[0, column]):
            num_cols.append(column)
    x_test = np.delete(x_test, num_cols, axis=1)
    # Replace elements of value -999 in 1st column of the data by the mean of
    # the non-NaN values of the columns
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
        x: Data to polynomially expand.
        degree: Degree of expansion.

    Returns:
        poly: Polynomially expanded data.
    """
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


def run_model(y, tx, model, gamma=0.05, max_iters=1000, lambda_=0):
    """Runs the chosen model with the appropriate parameters.

    Args:
        y:  A np array representing the labels.
        tx: A np array representing the input data.
        model: ML algorithm used.
          'gd': Gradient Descent,
          'sgd': Stochastic Gradient Descent,
          'lq': Least Squares,
          'ridge_reg': Ridge Regression,
          'log_reg': Logistic Regression
        initial_w: Initial weight vector used in Gradient Descent or Stochastic
          Gradient Descent.
        gamma: Step size used in Gradient Descent.
        max_iters: Nb of iterations.
        lambda_: Regularizer parameter.
    Returns:
        w: Weight vector w.
    """
    if model == 'gd':
        initial_w = np.zeros((tx.shape[1],))
        w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    elif model == 'sgd':
        initial_w = np.zeros((tx.shape[1],))
        w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
    elif model == 'lq':
        w, loss = least_squares(y, tx)
    elif model == 'ridge_reg':
        w, loss = ridge_regression(y, tx, lambda_)
    elif model == 'log_reg':
        initial_w = np.zeros((tx.shape[1],))
        w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    elif model == 'reg_log_reg':
        initial_w = np.zeros((tx.shape[1],))
        w, loss = reg_logistic_regression(
            y, tx, lambda_, initial_w, max_iters, gamma)
    return w


def build_k_indices(y, k_fold, seed):
    """Builds k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(K, y, x, model, gamma=0.05, max_iters=1000, lambda_=0,
                     logging=False, seed=0.8):
    """Applies cross validation to a given set of data.

    Args:
        K: Number of folds.
        y_batch: Labels of given dataset.
        tx_batch: Dataset.
        max_iters: Number of iterations run for each fold.
        gamma: step size parameter gamma.
        model: ML algorithm used.
          'gd': Gradient Descent,
          'sgd': Stochastic Gradient Descent,
          'lq': Least Squares,
          'ridge_reg': Ridge Regression,
          'log_reg': Logistic Regression
          'reg_log_reg': Regularized Logistic Regression
        initial_w: Initial weight vector used in Gradient Descent or Stochastic
          Gradient Descent.
        logging: Intermediary printing.
    Returns:
         A mean of the accuracies of the K different folds.
    """
    if model == 'log_reg' or model == 'reg_log_reg'
        # Make sure that the labels are 0 and 1 and not -1 and 1.
        y = np.where(y == -1, 0, y)
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
        w = run_model(y_tr, x_tr, model=model, gamma=gamma,
                      max_iters=max_iters, lambda_=lambda_)
        if model == 'log_reg' or model == 'reg_log_reg':
            acc = np.mean(
                predict_labels(w, x_te, threshold=0.5, logist=True,
                               negative_label=0, positive_label=1) == y_te)
            print(acc)
        else:
            acc = np.mean(predict_labels(w, x_te) == y_te)
        ws.append(w)
        accuracies.append(acc)
        if logging:
            print(acc)
    mu_acc = np.mean(accuracies)
    if logging:
        print("-------------- AVG : {}".format(mu_acc))
    return mu_acc


def forward_selection(y, tx, K):
    """Implements forward selection of the 2nd order.

    Takes the product of two columns, appends it to tx and assesses whether the
    resulting matrix is better for training our chosen model.

    Args:
        y: Labels to predict.
        tx: Data to iteratively augment with interaction terms.
        K: Number of folds used in the cross-validation.
    Returns:
        tx: Possibly augmented tx.
    """
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


def find_best_params(y, tx, K, max_degree=13):
    """Finds best degree and lambda given a certain dataset
    Args:
        tx: Data to polynomially expand and test.
        y: Labels.
        K: Number of folds used in the cross validation.
        max_degree: Polynomial expansion threshold.
    Returns: 
        best_degree_lambda: Tuple consisting of (Best_degree, Best_lambda).
    """
    degrees = np.arange(1, max_degree+1)
    lambdas = np.logspace(-4, -2, 10)
    lambdas = np.append(lambdas, 0)
    acc = []
    ind = []
    for d in degrees:
        for l in lambdas:
            ind.append((d, l))
            expanded = build_poly(tx, d)
            accuracy = cross_validation(
                K, y, expanded, model='ridge_reg', logging=True, lambda_=l, seed=0)
            acc.append(accuracy)

    best_degree_lambda, max_accuracy = ind[np.argmax(acc)], np.max(acc)
    print("Best parameters: for  polynomial degree ={}, lambda={} Acuracy:{}".format(
        best_degree_lambda[0], best_degree_lambda[1], max_accuracy))
    return best_degree_lambda
