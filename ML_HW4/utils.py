import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# visualize data 

def plot_svc(pred_ob, X, y, h=0.02, pad=0.25, plot_support_vector=False, plot_mis=False, sklearn=True, w=None):
    plt.figure(figsize=(12,8))
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    if sklearn:
        Z = pred_ob.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = pred_ob(np.c_[xx.ravel(), yy.ravel()], w)
    Z = Z.reshape(xx.shape)    
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    
    # Support vectors indicated in plot by vertical lines
    if plot_support_vector:
        sv = pred_ob.support_vectors_
        plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
        print('Number of support vectors: ', pred_ob.support_.size)
        
    if plot_mis:
        pred = pred_ob(X, w)
        label = y != pred
        plt.plot(X[label,0], X[label,1], 'rx')
        print('miss_classified data=', sum(label))
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    
def plot_all(svm, svm_auto_weight, logistic, X, y, h=0.02, pad=0.25):
    plt.figure(figsize=(12,8))
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    
    #svm sklearn
    coef = svm.coef_
    bias = svm.intercept_
    x = np.array([x_max, x_min])
    y = (-bias-coef[:,0]*x) / coef[:,1]
    plt.plot(x, y, 'k--', label='SVM sklearn')
    
    #svm autograd
    x = np.array([x_max, x_min])
    y = (-svm_auto_weight[0]-svm_auto_weight[1]*x) / svm_auto_weight[2]
    plt.plot(x, y, 'k:', label='SVM auto(yours)')
    
    #logistic
    coef = logistic.coef_
    bias = logistic.intercept_
    x_logi = np.array([x_max, x_min])
    y_logi = (-bias-coef[:,0]*x) / coef[:,1]
    plt.plot(x_logi, y_logi, 'k', label='logistic')
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
    
    
def vis_data(x, y = None, c='r'):
    plt.figure(figsize=(12, 8))
    if y is None: 
        y = [None] * len(x)
    for x_, y_ in zip(x, y):
        if y_ is None:
            plt.plot(x_[0], x_[1], 'o', markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c+'o' if y_ == -1 else c+'+')
    plt.grid(True)
    return


def vis_hyperplane(w, typ='k--'):
    
    lim0 = plt.gca().get_xlim()
    lim1 = plt.gca().get_ylim()
    m0, m1 = lim0[0], lim0[1]

    intercept0 = -(w[0] * m0 + w[-1])/w[1]
    intercept1 = -(w[0] * m1 + w[-1])/w[1]
    
    plt1, = plt.plot([m0, m1], [intercept0, intercept1], typ)

    plt.gca().set_xlim(lim0)
    plt.gca().set_ylim(lim1)
    
    
    return plt1


def gen_multimodal():
    x = np.random.multivariate_normal([-1,-1], [[1,0],[0,1]], 100)
    x = np.concatenate((x, np.random.multivariate_normal([1,1], [[1,0],[0,1]], 100)))
    x = np.concatenate((x, np.random.multivariate_normal([3,3], [[1,0],[0,1]], 50)))
    y = np.ones(250)
    y[100:200] = -1
    return x, y