import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *


def partition_data(x, y, i):
    "Merge data and labels"
    

    "Find indexes where the 23rd element of the data is 'i' (i = 0, 1, 2 and 3)"
    index_i = x[:, 22] == i
 
    "Split the data according to the indexes found"
    x_i = x[index_i]
    y_i = y[index_i]
    
    # print(x_i.shape)
    "-999s don't appear in 1st column following the indexes, so we don't take it in consideration in the following"
    "and we get rid of the then 21st column containing the 'i's"
   
    x_test = np.delete(x_i, 22, axis=1)
    # print(x_test.shape)
    "Remove columns where elements have value -999"
    num_cols=[]

    for column in range(x_test[0].size):
        if np.all(x_test[:,column]==-999):
            num_cols.append(column)
    x_test = np.delete(x_test,num_cols, axis=1)
    
    # Remove columns where elemnts are the same
    num_cols=[]
    for column in range(x_test[0].size):
        if np.all(x_test[:,column]==x_test[0, column]):
            num_cols.append(column)
    x_test = np.delete(x_test,num_cols, axis=1)

    "Concactenation of 1st column of the data and the rest with elements of value -999 removed"
    # print(-999 in x_i[:,0])
    

    "Replace elements of value -999 in 1st column of the data by the mean of "
    "Here, we can do a weighted mean instead of simply taking the mean "
    
    indices = x_test[:, 0] == -999
    # print( True in indices)
    x_test[indices,0] = np.mean(x_test[~indices, 0])
    print("Final shape : {} ".format(x_test.shape))
    return index_i, y_i, x_test 
   

def compute_corr(x_test):

    corr_mat = np.corrcoef(x_test, rowvar=False)
    return corr_mat

def build_poly_2(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x[:,1:],deg)]
    return poly


def visualize_corr(corr):
    fig, ax = plt.subplots(figsize=(corr.shape[0], corr.shape[1]))
    sns.heatmap(corr, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()


def standardize_data(x):
    """Standardize the original data set."""
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
            w: weight vector w.
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

    w = run_model(y_train, train, model = 'lq')
    return np.mean(predict_labels(w, test) == y_test)


def cross_validation(K, y_batch, tx_batch, model,  gamma=0.05, max_iters=100,  lambda_=0, logging=False, seed=0):
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
    N = tx_batch.shape[0]
    # We shuffle the data beforehand
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(N))
    shuffled_tx = tx_batch[shuffle_indices]
    shuffled_y = y_batch[shuffle_indices]

    fold_size = N//K  # euclidean divison => cast as int

    accuracies = []  # nb of accuracies to compute = K

    ws = []

    for i in range(0, N, fold_size):

        # Define the indexes of the training set
        start = i
        end = i + fold_size

        # Define training set
        train_tx = np.vstack((shuffled_tx[:start], shuffled_tx[end:N-1]))
        train_y = np.array(list(shuffled_y[:start])+list(shuffled_y[end:N-1]))

        # for each run of the model start with a random w
        rand_initial = [np.random.uniform(-1, 1)
                        for x in range(shuffled_tx.shape[1])]
        w = ridge_regression(train_y, train_tx, model, gamma,
                      rand_initial, max_iters, lambda_)

        # Test w
        ws.append(w)
        acc = np.mean(predict_labels(
            w, shuffled_tx[start: end]) == shuffled_y[start: end])
        accuracies.append(acc)

        if logging:
            print(acc)
    if logging:
        print("--------------")
    return np.mean(accuracies)

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_2(K, y, x, model,  gamma=0.05, max_iters=100,  lambda_=0, logging=True, seed=0.8):
    
    k_indices = build_k_indices(y,K,seed)
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
        
        w = run_model(y_tr, x_tr,model =model, lambda_ = lambda_)
        acc = np.mean(predict_labels(w, x_te) == y_te)
        ws.append(w)
        accuracies.append(acc)
        
        if logging:
            print(acc)
    if logging:
        print("--------------")
    return np.mean(accuracies)

def expand_features(x, degree):
    """ Expand features according to polynomial basis_acc.

    Args:
        x: A numpy array representing the data to expand.
        degree: Degree of polynomial expansion (i.e x^k).
    Returns:
        The same array with with expanded features.
    """
    array_expanded = []

    for column in x.T[1:]:
        array_expanded.append(
            np.vstack([column**k for k in range(1, degree+1)]).T)

    return np.hstack((np.ones((x.shape[0], 1)), np.concatenate(array_expanded,axis=1)))


# def look_for_best_degree_lambda(data, y,degree,lambda_start = 0, lambda_end):

#     range_gamma_start=0.02
#     range_gamma_end =0.15


#     degrees = np.arange(1,degree)

#     acc = []
#     ind=[]
#     for d in degrees: 
#         print(d)
#         acc.append(cross_validation(11, y0, expand_features(std_test0,d),model='ridge_reg',lambda_=0.5,logging=True))
#         ind.append((20,d))
        
#     print("Best parameters: lambda={}, polynomial degree ={}. Acuracy:{}".format(ind[np.argmax(acc)][0],ind[np.argmax(acc)][1],np.max(acc)))


def forward_selection(y, tx, K):

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


# def feature_combination(x, func_,columns):
#     additional_columns =  func_(a) for a in x[:,columns]

#     return np.array(additional_columns)
