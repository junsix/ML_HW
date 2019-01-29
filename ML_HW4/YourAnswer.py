import autograd.numpy as np
from autograd import grad
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#-*- coding: utf-8 -*-
""" 
HW 4.1
"""
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def d_sigmoid_without_autograd(x):
    d_sigmoid = 0
    ######################Write your code#######################
    # Instructions : Use the sigmoid function we defined above
    #                to get the derivative of sigmoid
    d_sigmoid = sigmoid(x) * ( 1 - sigmoid(x));

    
    ############################################################
    
    return d_sigmoid


def d_sigmoid_with_autograd(x):
    d_sigmoid = 0
    ######################Write your code#######################
    # Instructions : Use the autograd grad function
    #                to get the derivative of sigmoid
    d_nested_function = grad(lambda x: sigmoid(x))
    d_sigmoid = d_nested_function(x);
    
    ############################################################
    
    return d_sigmoid

"""
HW 4.2
"""
def score_function(x, y, w):
    '''
    INPUT: Feature vector (x) , class (y), and weight vector (w)
    Dimension:
    x: N*(d-1)
    w: d
    y: N
    OUTPUT: The score.
    '''
    score = np.zeros_like(y)
    ####################write your code#######################
    # Instructions : Implement the score function
    score = y*(np.dot(x,w[1:]) + w[:1])

    
    ##########################################################

    return score


def prediction_function(x, w):
    '''
    INPUT: Feature vector (x), and weight vector (w)
    Dimension:
    x: N*(d-1)
    w: d
    OUTPUT: The prediction.
    '''
    prediction = np.zeros(x.shape[0])
    ####################write your code#######################
    # Instructions : Implement the prediction function
    prediction = np.sign(x.dot(w[1:]) + w[:1])

    
    ##########################################################

    return prediction


def hinge_loss(x, y, w):
    '''
    INPUT: Feature vector (x) , class (y), and weight vector (w)
    Dimension:
    x: N*(d-1)
    w: d
    y: N
    OUTPUT: Hinge_loss vector.
    '''
    loss = np.zeros(x.shape[0])
    ####################write your code#######################
    # Instructions : Using the score function you implemented,
    #                compute the hinge loss
    loss = 1 - score_function(x, y, w)
    loss = np.clip(loss, 0, np.argmax(loss))
    ##########################################################

    return loss


def objective_function(w, x, y, C):
    '''
    Objective function. 

    INPUT: True labels (y), feature vector (x), weight vector (w) and constant (C)
    Dimension:
    x: N*(d-1)
    w: d
    y: N
    OUTPUT: Objective function value.
    '''
    obj = np.zeros(1)
    ####################write your code#######################
    # Instructions : Using the hinge_loss you implemented,
    #                compute the objective function value
    theta_1 = w[1:]
    obj = np.dot(theta_1, theta_1) / 2

    obj += np.sum(C * hinge_loss(x, y, w))
    
    
    ##########################################################
    
    return obj


def gridsearch(parameters, X, y):
    
    clf = None 
    ####################write your code###########################################
    # Instructions: Use GridSearchCV Function(Only use SVC(kernel='rbf') estimator) 
    #               in scikit-learn package to maximize accuracy!!.
    #               Set the number of folds to 10
    #               You should return the clf(classifier) correctly after fitting the data
    #
    # Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    clf = GridSearchCV(estimator=SVC(kernel='rbf'), 
                       param_grid=parameters, scoring='accuracy', cv=10)
    clf.fit(X,y)
    ##############################################################################
    return clf