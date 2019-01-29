import numpy as np
import operator
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
#-*- coding: utf-8 -*-

"""
KNN
"""

def euclideanDistance(targetX, dataSet):

    distances = 0
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to obtain euclideanDistance between one targetX and the other.
    #
    n = dataSet.shape[0]
    distances = np.zeros((n,))
    diff = np.copy(dataSet)
    for i in range(n):
        diff[i] -= targetX
        distances[i] = np.dot(diff[i], diff[i])
        distances[i] = np.sqrt(distances[i])
    
    # =========================================================================
    
    return distances

def getKNN(targetX, dataSet, labels, k):

    ## targetX : target data
    ## dataset : other data
    ## labels
    ## k : number of neighbors
    
    # compute euclidean distance
    distances = euclideanDistance(targetX,dataSet)
    closest_data = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Use the result of finding the distance between TargetX and other data,
    #               select the most out of k data closest to target data
    #
    bound = np.partition(distances, k - 1)[k - 1]
    n = dataSet.shape[0]
    sol = np.zeros((max(labels) + 1,))
    count = 0;

    for i in range(n):
        if distances[i] <= bound:
            sol[labels[i]] += 1
            if ( count < sol[labels[i]]):
                count = max(sol[labels[i]], count);
                closest_data = labels[i]
    # =========================================================================

    return closest_data

def predictKNN(targetX, dataSet, labels, k):
    
    ## targetX : target data
    ## dataset : other data
    ## labels
    ## k : number of neighbors
    
    n = targetX.shape[0]
    
    ## predicted_array : array for the predicted labels, having the same length as data.
    predicted_array = np.zeros((n,))
    for i in range(n):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Using the result of closest data from getKNN,
        #               put the predicted label in the predicted array.
        #
        predicted_array[i] = getKNN(targetX[i], dataSet, labels, k)
        #pass # erase this when implements
        
        # =========================================================================
    return predicted_array




"""
regularization
"""



def crossValidation_Ridge(lambdas, num_fold, X_train, y_train, X_test, y_test):

    y_train = y_train.values
    
    ## You should return these values/objects correctly
    MSE_set=[]
    best_MSE= 0.
    best_lambda = 0.
    test_MSE= 0.
    ridge = None
    
    for lamb in lambdas:
        MSE_fold = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: 
    #              Use K-fold Function in scikit-learn package to find best lambda of Ridge.
    #              Save the averaged MSE of each fold, and find best lambda
    #              And then fit the model using the best lambda selected above with the full X_train and y_train.
    #               
        
        # 1. Find MSE with K-fold using for loop ( http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html )            
        kf = KFold(n_splits=num_fold)

        for train_index, test_index in kf.split(X_train):
            ridge = Ridge(alpha=lamb)
            ridge.fit(X_train[train_index,:], y_train[train_index])
            pred = ridge.predict(X_train[test_index,:])
            MSE_fold += mean_squared_error(y_train[test_index], pred)
        
        # 2. Append the average of MSE of each fold in the MSE_set
        MSE_set.append(MSE_fold / num_fold)

    # 3. Find the best MSE and best lambda (hint : total number of MSE_set = len(lambdas))
    best_MSE = min(MSE_set)
    best_lambda = lambdas[MSE_set.index(best_MSE)]
    
    # 4. Using the best lambda, retrain Ridge() with full train data as 'ridge'
    ridge = Ridge(alpha=best_lambda)
    ridge.fit(X_train, y_train)
    
    # 5. Save the MSE of retrained model on the test data as 'test_MSE'
    pred = ridge.predict(X_test)
    test_MSE = mean_squared_error(y_test, pred)
    
    # ============================================================
    return MSE_set, best_MSE, best_lambda, test_MSE,ridge

def crossValidation_Lasso(lambdas, num_fold, X_train, y_train, X_test, y_test):

    y_train = y_train.values
    
    ## You should return these values/objects correctly
    MSE_set=[]
    best_MSE=0.
    best_lambda = 0.
    test_MSE=0.
    lasso = None
    
    for lamb in lambdas:
        MSE_fold = 0.
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: 
    #              Use K-fold Function in scikit-learn package to find the best lambda of Lasso.
    #              Save the averaged MSE of each fold, and find the best lambda
    #              And then fit the model using the best lambda selected above with the full X_train and y_train.
    #               

        # 1. Find MSE with K-fold using for loop ( http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html )
        kf = KFold(n_splits=num_fold)
        for train_index, test_index in kf.split(X_train):
            lasso = Lasso(alpha=lamb)
            lasso.fit(X_train[train_index,:], y_train[train_index])
            pred = lasso.predict(X_train[test_index,:])
            MSE_fold += mean_squared_error(y_train[test_index], pred)
        
        # 2. Append the average of MSE of each fold in the MSE_set
        MSE_set.append(MSE_fold / num_fold)
    
    # 3. Find the best MSE and best lambda (hint : total number of MSE_set = len(lambdas))
    best_MSE = min(MSE_set)
    best_lambda = lambdas[MSE_set.index(best_MSE)]

    
    # 4. Using the best lambda, retrain Lasso() with full train data as 'lasso'
    lasso = Lasso(alpha=best_lambda)
    lasso.fit(X_train, y_train)
    
    
    # 5. Save the MSE of retrained model on the test data as 'test_MSE'
    pred = lasso.predict(X_test)
    test_MSE = mean_squared_error(y_test, pred)

    # ============================================================
    return MSE_set, best_MSE, best_lambda, test_MSE, lasso