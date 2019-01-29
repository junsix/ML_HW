#-*- coding: utf-8 -*-
import numpy as np
import operator


def cost_naive(X, y, theta):
    # Compute cost for linear regression by using loop
    # J = cost_navie(X, y, theta) computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y
    
    # some useful values
    n = len(X)
    
    # You need to return this value correctly:
    J = 0
    
    # ====================== YOUR CODE HERE ==  ====================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost. Access each data in a loop.

    sum = 0;
    for i in range(n):
        #sum += (y[i] - X[i][0] * theta[0] - X[i][1] * theta[1]) ** 2
        sum += (y[i] - X[i].dot(theta)) ** 2

    J = sum / (2 * n);
    # ============================================================
    return J


def cost_vectorized(X, y, theta):
    # Compute cost for linear regression by vectorized computation
    # J = cost_vectorized(X, y, theta) computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y

    # some useful values
    n = len(X)

    # You need to return this value correctly:
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost, and consider vector multiplication.

    temp = y - X.dot(theta)
    J = temp.dot(temp) / (2 * n)

    # ============================================================
    return J


def gradient_descent_func_naive(X, y, theta_, alpha, num_iters):
    # Performs gradient descent to learn theta by using loop
    # theta = gradient_descent_func_naive(X, y, theta, alpha, num_iters) updates theta by 
    # taking num_iters gradient steps with learning rate alpha
    
    # Initialize
    theta = theta_.copy()
    J_his = np.zeros((num_iters,))
    T_his = np.zeros((num_iters,2))
    for i in range(num_iters):
        T_his[i] = theta

        ### ========= YOUR CODE HERE ============
        # Instructions: Perform a single gradient step on the parameter vector theta.
        # Access each data in a loop.
        update = 0
        n = len(X)
        for i in range(n):
            #update += (y[i] - X.dot(theta)[i]) * X[i]
            update += (y[i] - X[i].dot(theta)) * X[i]
        update *= alpha / n
        theta += update
        
        ### =====================================
        J_his[i] = cost_naive(X, y, theta)
    return theta, J_his, T_his


def gradient_descent_func_vectorized(X, y, theta_, alpha, num_iters):
    # Performs gradient descent to learn theta by vectorized computation
    # theta = gradient_descent_func_vectorized(X, y, theta, alpha, num_iters) updates theta by 
    # taking num_iters gradient steps with learning rate alpha
    
    # Initialize
    theta = theta_.copy()
    J_his = np.zeros((num_iters,))
    T_his = np.zeros((num_iters,2))
    for i in range(num_iters):
        T_his[i] = theta

        ### ========= YOUR CODE HERE ============
        # Instructions: Perform a single gradient step on the parameter vector theta.
        # You should consider vector multiplication.
        n = len(X)
        update = (np.transpose(X).dot(X)).dot(theta) - np.transpose(X).dot(y)
        update /= n
        theta -= alpha * update
        
        ### =====================================
        J_his[i] = cost_vectorized(X, y, theta)
    return theta, J_his, T_his


def ols_func(X, y):
    # Find empirical risk minimizer
    theta = np.zeros(X.shape[1])
    
    ### ========= YOUR CODE HERE ============
    # Instructions: Compute theta by Ordinary Least Squares method
    theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)

    ### =====================================
    
    return theta


def perceptron(X, th):
    # Compute perceptron function 
    # h = perceptron(X, th) decides whether its input belongs to a class or not
    # You need to return the following variable correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute perceptrons of each value of X (X can be a matrix,
    #               vector or scalar).

    for i in range(X.shape[0]):
        p[i] = (X[i].dot(th) > 0)

    # =============================================================

    return p


def perceptron_cost(X, y, th):
    # COSTFUNCTION Compute cost for perceptron
    #   J = COSTFUNCTION(theta, X, y) computes the cost
    #   using theta as the parameter for perceptron

    # Initialize some useful values
    n = len(y)

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    h = perceptron(X, th)
    J -= (y - h).dot(X.dot(th))
    # for i in range(n):
    #    dot = X[i].dot(th)
    #    h = 0
    #    if dot > 0:
    #        h = 1
    #    J -= (y[i] - h) * dot

    J /= n

    # =============================================================
    
    return J


def perceptron_rule(X, y, th):
    # Compute gradients of perceptrons
    # Initialize some useful values
    n = len(y)
    
    # You need to return the following variable correctly 
    grad = np.zeros(th.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute perceptrons of each value of X (X can be a matrix,
    #               vector or scalar).
    h = perceptron(X, th)
    grad -= (y - h).dot(X)
    #for i in range(n):
    #    dot = X[i].dot(th)
    #    h = 0
    #    if dot > 0:
    #       h = 1
    #    grad -= (y[i] - h) * X[i]
    grad /= n
    
    # =============================================================
    return grad


def sigmoid(z):
    #SIGMOID Compute sigmoid function
    #   J = SIGMOID(z) computes the sigmoid of z.
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).
    
    g = 1 / (1 + np.exp(-z))
    
    # =============================================================
    
    return g



def logistic_regression_cost(X, y, th, _lambda=0):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(X, y, th) computes the cost using theta as the
    #   parameter for logistic regression

    # Initialize some useful values
    n = len(y)

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.

    for i in range(n):
        h = sigmoid(X[i].dot(th))
        J -= y[i] * np.log(h) + (1 - y[i]) * np.log(1 - h)
    J /= n

    # =============================================================

    return J


def logistic_regression_gradient(X, y, th, _lambda=0):
    # Initialize some useful values
    n = len(y)
    
    # You need to return the following variables correctly
    grad = np.zeros(th.shape)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    for i in range(n):
        h = sigmoid(X[i].dot(th))
        grad -= (y[i] - h) * X[i]
    grad /= n
    # =============================================================
    
    return grad


def logistic_regression_predict(th, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic 
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a 
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    
    n = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(n)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters. 
    #               You should set p to a vector of 0's and 1's
    #

    for i in range(n):
        h = sigmoid(X[i].dot(th))
        if h >= 0.5:
            p[i] = 1
    # =========================================================================
    
    return p

