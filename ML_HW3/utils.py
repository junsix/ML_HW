import matplotlib.pyplot as plt
from YourAnswer import predictKNN
import numpy

def plotData(data):
    fig, ax = plt.subplots(figsize=(8,5))
    results_accepted = data[data.accepted == 1]
    results_rejected = data[data.accepted == 0]
    ax.scatter(results_accepted.test1, results_accepted.test2, marker='+', c='b', s=40)
    ax.scatter(results_rejected.test1, results_rejected.test2, marker='o', c='r', s=30)
    return ax

def vis_decision_boundary(x_tra, y_tra, k, typ='k--'):
    ax = plt.gca()

    lim0 = plt.gca().get_xlim()
    lim1 = plt.gca().get_ylim()
    
    x_ = numpy.linspace(lim0[0], lim0[1], 100)
    y_ = numpy.linspace(lim1[0], lim1[1], 100)
    xx, yy = numpy.meshgrid(x_, y_)
    
    pred = predictKNN(numpy.concatenate([xx.ravel()[:,None], yy.ravel()[:,None]], axis=1), x_tra, y_tra, k)
    
    ax.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.4)

    ax.set_xlim(lim0)
    ax.set_ylim(lim1)
    
def vis_coef(lambdas, coefs, method=''):
    plt.figure()
    ax = plt.gca()
    ax.plot(lambdas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.axis('tight')
    plt.xlabel('lambda')
    plt.ylabel('weights')
    plt.title('{} coefficients as a function of the regularization'.format(method))
    plt.show()
    
def vis_mse(lambdas, MSE_set, best_lambda, best_mse):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(lambdas, MSE_set)
    ax.scatter(best_lambda, best_mse, s=100, c='r', marker='x')
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    ax.text(x=best_lambda+30, y=best_mse+5000,s='MSE : {:.2f}'.format(best_mse))
    plt.axis('tight')
    plt.xlabel('lambda')
    plt.ylabel('Mean Squared Error')
    plt.title('Validation set MSE')
    plt.show()
    