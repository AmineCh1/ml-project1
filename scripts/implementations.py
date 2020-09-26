
'''
All functions should return only last_loss,last_w, unlike the labs where we kept track of all iterations in two arrays.
'''

'''
Used in 
'''

def compute_loss(y,tx,w):
    #TODO : Implement loss function (might change often)

    raise NotImplementedError

def compute_gradient(y,tx,w):
    #used in GD and SGD
    #TODO:  computes gradient for each iteration of least_squares_GD/least_squares_SGD
    
    raise NotImplementedError

def compute_subgradient(y,tx,w):
    #In case we want to play with non-differentiable functions
    #TODO : computes gradient for locally non-differentiable functions (i.e |x|)

    raise NotImplementedError

def least_squares_GD(y,tx,initial_w, max_iters,gamma):
    #TODO : linear regression using gradient descent

    raise NotImplementedError

def least_squares_SGD(y,tx,initial_w, max_iters,gamma):
    #TODO : Linear regression using stochastic gradient descent
    
    raise NotImplementedError

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