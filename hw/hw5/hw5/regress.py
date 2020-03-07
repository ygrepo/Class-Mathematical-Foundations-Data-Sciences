import numpy as np
_beta = np.array([9,1,1,1,1])

"""
Returns the tuple train_X,train_y,valid_X,valid_y containing
the training and validation sets.
"""
def get_data() :
    global _beta
    np.random.seed(1234)
    num_train, num_valid = 50,20
    n = num_train+num_valid
    X = np.zeros((n,5))
    X[:,0] = 1
    X[:,1] = X[:,0]+np.random.randn(n)
    for i in range(2,5) :
        X[:,i] = X[:,i-1] + np.random.randn(n)*0.01
    y = np.dot(X,_beta) + np.random.randn(n)
    return X[:num_train,:],y[:num_train],X[num_train:,:],y[num_train:]

def main() :
    train_X,train_y,valid_X,valid_y = get_data()
    
if __name__ == "__main__" :
    main()
