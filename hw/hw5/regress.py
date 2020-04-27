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

def least_squares(X,y) :
    be,res,rank,s = np.linalg.lstsq(X,y)
    return be,s

def square_loss(b,X,y) :
    return np.linalg.norm(np.dot(X,b)-y)**2

def ridge(X,y,lam) :
    m = X.shape[1]
    A = np.concatenate((X,np.sqrt(lam)*np.eye(m)),axis=0)
    b = np.concatenate((y,np.zeros(m)))
    be,res,rank,s = np.linalg.lstsq(A,b)
    return be,s

def print_results(name,b,train_X,train_Y,valid_X,valid_Y,sing) :
    print('%s: beta:'%name,b)
    print('%s: Singular values:'%name,sing)
    print('%s: Training Square Loss: %f'%(name,square_loss(b,train_X,train_Y)))
    print('%s: Validation Square Loss: %f'%(name,square_loss(b,valid_X,valid_Y)))
    print('%s: Correlation Matrix:'%name,np.corrcoef(train_X[:,1:].T))

def main() :
    train_X,train_y,valid_X,valid_y = get_data()
    b_ls,sing_ls = least_squares(train_X,train_y)
    b_r,sing_r = ridge(train_X,train_y,0.5)
    print_results('LS',b_ls,train_X,train_y,valid_X,valid_y,sing_ls)
    print_results('Ridge',b_r,train_X,train_y,valid_X,valid_y,sing_r)
    
if __name__ == "__main__" :
    main()
