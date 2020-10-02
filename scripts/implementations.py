import numpy as np
'''
All functions should return only last_loss, last_w, unlike the labs where we
kept track of all iterations in two arrays.

TODO: FINAL submission needs to be according to the instructions in the project
description.

Nevertheless, we keep track of the intermediate w and losses for visualization
purposes. 
'''



def compute_loss_MSE(e):
    """Computes the mean squared error from an error vector."""
    return 1/2 * np.mean(e**2)
    
    
def compute_loss_MAE(e):
    """Computes the mean absolute error from an error vector."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, function='MSE'):
    """Calculate the loss.
    
    You can calculate the loss using MSE or MAE.
    """
    e = y - tx.dot(w)
    if function == 'MAE':
        return compute_loss_MAE(e)
    else:
        if function != 'MSE':
            print("Loss function unknown. Switching to MSE but please double
                  check")
        return compute_loss_MSE(e)

    
def compute_gradient(y, tx, w):
    """Computes the gradient of the weight vector in MSE for linear regression.
    
    Returns the error as well as the gradient.
    """
    N = len(y)
    e = y - tx @ w
    return (-1 / N) * tx.T @ e, e


def compute_subgradient(y, tx, w):
    #In case we want to play with non-differentiable functions
    #TODO : computes gradient for locally non-differentiable functions (i.e |x|)
    raise NotImplementedError


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Computes a least squares model using gradient descent.
    
    Args:
        y: A numpy array representing the output variable.
        tx: A numpy array representing the transpose matrix of input variable X.
        initial_w: A numpy array representing the weights of each feature.
        max_iters: An integer specifying the maximum number of iterations for
            convergence.
        gamma: A float number greater than 0 used as a learning rate.
        
    Returns:
        A tuple with intermediate losses and weights respectively.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter_ in range(max_iters):
        grad, e = compute_gradient(y, tx, w)
        loss = compute_loss_MSE(e)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        # Logging
        print(f"Gradient Descent({iter_}/{max_iters - 1}): loss={loss}, w={w}")
    return losses[-1], ws[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Computes a least squares model using stochastic gradient descent."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter_ in range(max_iters):
        # This for loop has one iteration.
        for min_batch_y, min_batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, e = compute_gradient(min_batch_y, min_batch_tx, w)
            loss = compute_loss_MSE(e)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
            # Logging
            print(f"SGD({iter_}/{max_iters - 1}): loss={loss}, w={w}")
    return losses[-1], ws[-1]


def least_squares(y,tx):
    """Calculate the least squares using the normal equations."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return loss, w

                  
def ridge_regression(y,tx,lambda_):
    #TODO : Ridge regression using normal equations
    raise NotImplementedError

                  
def logistic_regression(y,tx,initial_w,max_iters,gamma):
    #TODO : Logistic regression using gradient descent or SGD
    raise NotImplementedError

                  
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #TODO : Regularized logistic regression using gradient descent 
    #or SGD
    raise NotImplementedError