import os

import matplotlib.pyplot as plt
import numpy as np


def calc_true_error(x1, x2):
    ''' eigenvecs could converge to u or -u - both are valid eigvecs.
    The function should output the L2 norm of (x1 - x2)
    If x1 = u and x2 = -u, we still want the function to output 0 error'''
    return np.linalg.norm(np.abs(x1) - np.abs(x2))


def eigen_iteration(A, x0, alpha, max_iter=50, thresh=1e-3):
    '''A - nxn symmetric matrix
       x0 - np.array of dimension n which is the starting point
       alpha - learning rate parameter
       max_iter - number of iterations to perform
       thresh - threshold for stopping iteration
       
       stopping criteria: can stop when ||x[k] - x[k-1]||_2 <= thresh or when it hits max_iter
       
       return: 
       relative_error: array with ||x[k] - x[k-1]||_2  
       true_error: array with || |x[k]| - |u_1| ||_2 where u_1 is first eigenvector
       '''

    assert ((A.transpose() == A).all())  # asserting A is symmetric
    assert (A.shape[0] == len(x0))

    w, v = np.linalg.eigh(A)
    true_u1 = v[:, 0]  # np array with the first eigenvector of A
    relative_error = []
    true_error = []
    x_cur = x0.copy()
    iteration = 0
    while True:
        x_next = x_cur + alpha * np.matmul(-2 * A, x_cur)
        x_next = np.divide(x_next, np.linalg.norm(x_next))

        rel_error = np.linalg.norm(x_cur - x_next)
        if rel_error <= thresh:
            print("Convergence in {} iterations, alpha:{},\
             init_point_norm={}".format(iteration, alpha, np.linalg.norm(x0)))
            print("True u1:{}, computed u1:{}, rel_error:{}, true_error:{}"
                  .format(true_u1, x_next, rel_error, calc_true_error(x_cur, true_u1)))
            break
        iteration += 1
        if iteration >= max_iter:
            print("Maximum iteration exceeded!")
            print("True u1:{}, computed u1:{}, true_error:{}, alpha:{}"
                  .format(true_u1, x_next, rel_error, calc_true_error(x_cur, true_u1), alpha))
            break
        relative_error.append(rel_error)
        true_error.append(calc_true_error(x_cur, true_u1))
        x_cur = x_next

    ## fill in code to do do your projected gradient ascent
    ## append both the list with the errors

    return relative_error, true_error


def load_matrices(path):
    A_list = []
    for x in os.listdir(path):
        A_list.append(np.genfromtxt(os.path.join(path, x), delimiter=','))
    return A_list


def plot(rel_errors, true_errors, alpha_array):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    for i, alpha in enumerate(alpha_array):
        axes[0].semilogy(rel_errors[i], '--o', label=str(alpha), alpha=0.8)
        axes[1].semilogy(true_errors[i], '--o', label=str(alpha), alpha=0.8)

    axes[0].set_xlabel('Iteration Number')
    axes[0].set_ylabel('Error')
    axes[0].set_title('Relative error b/w iterations')
    axes[1].set_xlabel('Iteration Number')
    axes[1].set_title('Error with true vector')
    plt.legend()
    # plt.show()
    return plt


def main():
    np.random.seed(123)
    alpha_array = 10 ** np.array([-1.0, 0, 1, 2, 8])
    # max_iter = 2
    max_iter = 200
    matrix_list = load_matrices('./data')

    for A in matrix_list:
        init_point = np.random.randn(A.shape[0])
        true_errors = list()
        rel_errors = list()

        for alpha in alpha_array:
            t1, t2 = eigen_iteration(A, init_point, alpha=alpha, max_iter=max_iter)
            rel_errors.append(t1)
            true_errors.append(t2)

        plt = plot(rel_errors, true_errors, alpha_array)
        plt.savefig('random_init_'+str(A.shape[0]) + '.pdf', bbox_inches = 'tight')
        plt.show()


def main_test():
    np.random.seed(1234)
    alpha_array = 10 ** np.array([-1.0, 0, 1, 2, 8])
    max_iter = 2000
    A = np.diag((1, 2, 3))
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    print(A)
    init_point = np.random.randn(A.shape[0])
    true_errors = list()
    rel_errors = list()
    for alpha in alpha_array:
        t1, t2 = eigen_iteration(A, init_point, alpha=alpha, max_iter=max_iter)
        rel_errors.append(t1)
        true_errors.append(t2)
    plt = plot(rel_errors, true_errors, alpha_array)
    plt.show()


if __name__ == "__main__":
    # main_test()
    main()
