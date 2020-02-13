import numpy as np
import matplotlib.pyplot as plt
import os


def calc_true_error(x1, x2):
    ''' eigenvecs could converge to u or -u - both are valid eigvecs. 
    The function should output the L2 norm of (x1 - x2)
    If x1 = u and x2 = -u, we still want the function to output 0 error'''
    
   
    return np.linalg.norm(x1 - x2)


def eigen_iteration(A, x0, alpha, max_iter = 50, thresh = 1e-3):
    '''A - nxn symmetric matrix
       x0 - np.array of dimension n which is the starting point
       alpha - learning rate parameter
       max_iter - number of iterations to perform
       thresh - threshold for stopping iteration
       
       stopping criteria: can stop when ||x[k] - x[k-1]||_2 <= thresh or when it hits max_iter
       
       return: 
       relative_error: array with ||x[k] - x[k-1]||_2  
       true_error: array with ||x[k] - u_1 ||_2 where u_1 is first eigenvector
       '''
    
    assert( (A.transpose() == A).all() ) #asserting A is symmetric
    assert( A.shape[0] == len(x0) )
    
    
    true_u1 = # np array with the first eigenvector of A
    
    relative_error = []
    true_error = []
    
    ## fill in code to do do your projected gradient ascent
    ## append both the list with the errors

    return relative_error, true_error


def load_matrices(path):
    A_list = []
    for x in os.listdir(path):
        A_list.append(np.genfromtxt( os.path.join(path, x), delimiter=',' ) )
    return A_list


alpha_array = 10**np.array([-1.0, 0, 1, 2, 8])
max_iter = 200

matrix_list = load_matrices('./data')


for A in matrix_list:
    
    init_point = np.random.randn(A.shape[0])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12, 4))

    for alpha in alpha_array:
        t1, t2 = eigen_iteration(A, init_point, alpha = alpha, max_iter = max_iter )
        axes[0].semilogy(t1, '--o', label = str(alpha), alpha = 0.8)
        axes[1].semilogy(t2, '--o', label = str(alpha), alpha = 0.8)

    axes[0].set_xlabel('Iteration Nnmber')
    axes[0].set_ylabel('Error')
    axes[1].set_xlabel('Iteration Nnmber')
    axes[0].set_title('Relative error b/w iterations')
    axes[1].set_title('Error with true vector')
    plt.legend()
    plt.savefig('random_init_'+str(A.shape[0]) + '.pdf', bbox_inches = 'tight')
    plt.show()
    
    


