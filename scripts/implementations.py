import numpy as np
'''
All functions should return only last_loss,last_w, unlike the labs
 where we kept track of all iterations in two arrays.

'''



def compute_loss_MSE(y,tx,w):
    
    N = len(y)
    e  = y -tx@w
   
    return (1/(2*N))*e.T@e 

    
def compute_loss_MAE(y,tx,w):
   
    N = len(y)
    e = y - tx@w

    return (1/N)*np.sum(np.abs(e))

 

def compute_gradient(y,tx,w):
    #used in GD and SGD

    N = len(y)
    e = y -tx@w

    return (-1/N)*tx.T@e


def compute_subgradient(y,tx,w):
    #In case we want to play with non-differentiable functions
    
    N  = len(y)
    e = y -tx@w 
    
    return (-1/N)*np.sign(e)@tx

    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    ATTENTION : COPIED FROM PROVIDED FUNCTIONS IN LAB 02 . Don't know 
    if this is okay ....
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

            
def least_squares_GD(y,tx,initial_w, max_iters,gamma):

    ws = [initial_w]
    losses = []
    w = initial_w

    for iter_ in range(max_iters):
        
        grad = compute_gradient(y,tx,w)
        loss = compute_loss_MSE(y,tx,w)
        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)

#         print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#               bi=iter_, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
   
    return losses[-1],ws[-1]


def least_squares_SGD(y,tx,initial_w, max_iters,gamma):
    
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    for iter_ in range(max_iters):

        grad = [0]*tx.shape[1]
        
        for min_batch_y, min_batch_tx in batch_iter(y,tx,batch_size= 1,num_batches = 1):
            grad = grad + compute_gradient(min_batch_y,min_batch_tx,w)
        loss = compute_loss_MSE(y,tx,w)

        w = w - (gamma*grad)

        ws.append(w)
        losses.append(loss)
        
#         print("Gradient Descent({bi}/{ti}): loss={l}".format(
#               bi=iter_, ti=max_iters - 1, l=loss))
        
    return losses[-1],ws[-1]

def least_squares(y,tx):
    #TODO : Least squares regression using normal equations
    
    raise NotImplementedError

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