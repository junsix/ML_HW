from sklearn.datasets import make_blobs
import numpy as np

def assign(c, x):
    
    Z = np.zeros(x.shape[0])
    # ====================== YOUR CODE HERE ======================
    # Instructions: Update the cluster membership. 
    #               You should cluster the data to the closest centroid point.
    #               Z should be the membership array of each data point.
    #               If you need, you can use for loop here. There will be no penalty.
    #               No for loop is recommended though.
    
    for i in range(x.shape[0]):
        T =  np.zeros(c.shape[0])
        for j in range(c.shape[0]):
            T[j] = np.sqrt((x[i]-c[j]).dot(x[i]-c[j]))
        Z[i] = np.argmin(T)
    
    # ============================================================
    
    return Z

def update_centroid(Z, c, x, K):
    for kk in range(K):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Update the cluster centroid. 
    #               New centroids c should be the average of the data points
    #               in each cluster.
    #               Careful with centroids of no data, otherwise they would be nan.
        sum = 0
        count = 0
        for i in range(x.shape[0]):
            if (Z[i] == kk):
                sum += x[i]
                count += 1
        sum /= count
        c[kk] = sum
        #pass
        
    # ============================================================
                 
    return c

def kmeans(x, K, max_iter=100):
    n_dim = x.shape[1]
    np.random.seed(0)
    c = np.random.rand(K, n_dim) * 1.5
    Z = np.zeros(x.shape[0])
    
    for _ in range(max_iter):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Using the assign & update_centroid function,
    #               repeats the process until convergence  
        Z = assign(c, x)
        update_centroid(Z, c, x, K)
        #pass
    # ============================================================
    
    return c, Z

def K_validate(x, K_range):

    cost = []
    
    for kk in K_range:
    # ====================== YOUR CODE HERE ======================
    # Instructions: Using the kmeans function,
    #               repeats the process for various K.
    #               Compute the cost of each process.
        c, Z = kmeans(x, kk, max_iter=100)
        sum = 0
        for i in range(x.shape[0]):
            sum += np.sqrt((x[i]-c[int(Z[i])]).dot(x[i]-c[int(Z[i])]))
        cost.append( sum / x.shape[0])
        #pass
    # ============================================================
    
    return cost

# SVD

def SVD(X):
    e_val = np.zeros(X.shape[1])
    e_vec = np.zeros((X.shape[1], X.shape[1]))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Find the right singular vectors.
    #               From the sample covariance matrix of X (X^T X),
    #               compute the eigenvectors and eigenvalues.
    #       Hint : numpy.linalg.eig
    cov = np.transpose(X).dot(X)
    e_val, e_vec = np.linalg.eig(cov)
    # ============================================================
    
    return e_val, e_vec

# get PCA

def PCA(X, e_vec):
    Z = np.zeros((X.shape[0],e_vec.shape[1]))
    # ====================== YOUR CODE HERE ======================
    # Instructions: return 'Z' as PCA of X

    Z = X.dot(e_vec)

    # ============================================================
    return Z