import os

import matplotlib.pyplot as plt
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_true_error(x1, x2):
    ''' eigenvecs could converge to u or -u - both are valid eigvecs.
    The function should output the L2 norm of (x1 - x2)
    If x1 = u and x2 = -u, we still want the function to output 0 error'''
    return 1 - abs(np.cos(angle_between(x1, x2)))
    # return np.linalg.norm(np.abs(x1) - np.abs(x2))


def eigen_iteration(A, x0, alpha, max_iter=50, thresh=1e-5):
    '''A - nxn symmetric matrix
       x0 - np.array of dimension n which is the starting point
       alpha - learning rate parameter
       max_iter - number of iterations to perform
       thresh - threshold for stopping iteration

       stopping criteria: can stop when |lambda[k] - lambda[k-1]| <= thresh or when it hits max_iter

       return:
       relative_error_eigvec: array with ||x[k] - x[k-1]||_2
       true_error_eigvec: array with ||x[k] - u_1 ||_2 where u_1 is first eigenvector
       relative_error_eigval: array with |lambda[k] - lambda[k-1] |
       true_error_eigval: array with |lambda[k] - lambda_1|

       x[k] is your estimated max eigenvec at iteration k and lambda[k] is your estimated max eigenvalue at iteration k.
       lambda_1 is the max eigenvalue of A and u_1 is the corresponding eigvec.
       '''

    assert ((A.transpose() == A).all())  # asserting A is symmetric
    assert (A.shape[0] == len(x0))

    w, v = np.linalg.eigh(A)
    true_lam = w[w.size - 1]  # fill in your code to find max eigenvalue of A
    true_u1 = v[:, v.shape[1] - 1]  # np array with the first eigenvector of A
    relative_errors_eigvec = list()
    true_errors_eigvec = list()
    relative_errors_eigval = list()
    true_errors_eigval = list()
    curr_eigvec = x0.copy()
    iteration = 1
    while True:
        next_eigv = curr_eigvec + alpha * np.matmul(-2 * A, curr_eigvec)
        next_eigv = unit_vector(next_eigv)

        rel_eigvec_error = np.linalg.norm(next_eigv - curr_eigvec)
        relative_errors_eigvec.append(rel_eigvec_error)
        true_eigvec_error = calc_true_error(true_u1, next_eigv)
        true_errors_eigvec.append(true_eigvec_error)

        eigval_prev = curr_eigvec.T.dot(np.matmul(A, curr_eigvec))
        eigval_next = next_eigv.T.dot(np.matmul(A, next_eigv))
        rel_eigval_error = abs(eigval_next - eigval_prev)
        relative_errors_eigval.append(rel_eigval_error)
        true_eigval_error = abs(true_lam - eigval_next)
        true_errors_eigval.append(true_eigval_error)

        if rel_eigval_error <= thresh:
            print("Convergence in {} iterations, alpha:{},\
             init_point_norm={}".format(iteration, alpha, np.linalg.norm(x0)))
            print("True u1:{}, computed u1:{}, rel_error:{}, true_error:{}"
                  .format(true_u1, next_eigv, rel_eigvec_error, true_eigvec_error))
            print("True max.eigenval:{}, computed max_eigval:{}, rel_error:{}, true_error:{}"
                  .format(true_lam, eigval_next, rel_eigval_error, true_eigval_error))
            break
        iteration += 1
        if iteration >= max_iter:
            print("Maximum iteration exceeded!")
            print("True u1:{}, computed u1:{}, rel_error:{}, true_error:{}"
                  .format(true_u1, next_eigv, rel_eigvec_error, true_eigvec_error))
            print("True max.eigenval:{}, computed max_eigval:{}, rel_error:{}, true_error:{}"
                  .format(true_lam, eigval_next, rel_eigval_error, true_eigval_error))
            break

        curr_eigvec = next_eigv

    ## fill in code to do do your projected gradient ascent
    ## append both the list with the errors

    return relative_errors_eigvec, true_errors_eigvec, relative_errors_eigval, true_errors_eigval


def load_matrices(path):
    A_list = []
    for x in os.listdir(path):
        A_list.append(np.genfromtxt(os.path.join(path, x), delimiter=','))
    return A_list


def plot(rel_errors_eigvec, true_errors_eigvec, rel_errors_eigval, true_errors_eigval, alpha_array):
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for i, alpha in enumerate(alpha_array):
        axes[0, 0].semilogy(rel_errors_eigvec[i], '--o', label=str(alpha), alpha=0.8)
        axes[0, 1].semilogy(true_errors_eigvec[i], '--o', label=str(alpha), alpha=0.8)
        axes[1, 0].semilogy(rel_errors_eigval[i], '--o', label=str(alpha), alpha=0.8)
        axes[1, 1].semilogy(true_errors_eigval[i], '--o', label=str(alpha), alpha=0.8)

    axes[0, 0].set_ylabel('Iteration Number')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].set_title('Relative error b/w iterations')

    axes[0, 1].set_ylabel('Iteration Number')
    axes[0, 1].set_title('True Error of eigenvector')

    axes[1, 0].set_xlabel('Iteration Number')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_title('Relative Error of eigenvalue')

    axes[1, 1].set_xlabel('Iteration Number')
    axes[1, 1].set_title('True Error of eigenvalue')
    plt.legend()
    return plt


def main():
    np.random.seed(123)
    alpha_array = 10 ** np.array([-1.0, 0, 1, 2, 8])
    # max_iter = 2
    max_iter = 200
    matrix_list = load_matrices('./data')

    for A in matrix_list:
        init_point = np.random.randn(A.shape[0])
        true_errors_eigvec = list()
        rel_errors_eigvec = list()
        true_errors_eigval = list()
        rel_errors_eigval = list()
        for alpha in alpha_array:
            t1, t2, t3, t4 = eigen_iteration(A, init_point, alpha=alpha, max_iter=max_iter)
            rel_errors_eigvec.append(t1)
            true_errors_eigvec.append(t2)
            rel_errors_eigval.append(t3)
            true_errors_eigval.append(t4)
        plt = plot(rel_errors_eigvec, true_errors_eigvec, rel_errors_eigval, true_errors_eigval, alpha_array)
        plt.savefig('random_init_' + str(A.shape[0]) + '.pdf', bbox_inches='tight')
        plt.show()


def main_test():
    np.random.seed(1234)
    alpha_array = 10 ** np.array([-1.0, 0, 1, 2, 8])
    max_iter = 2000
    A = np.diag((1, 2, 3))
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    print(A)
    init_point = np.random.randn(A.shape[0])
    true_errors_eigvec = list()
    rel_errors_eigvec = list()
    true_errors_eigval = list()
    rel_errors_eigval = list()
    for alpha in alpha_array:
        t1, t2, t3, t4 = eigen_iteration(A, init_point, alpha=alpha, max_iter=max_iter)
        rel_errors_eigvec.append(t1)
        true_errors_eigvec.append(t2)
        rel_errors_eigval.append(t3)
        true_errors_eigval.append(t4)
    plt = plot(rel_errors_eigvec, true_errors_eigvec, rel_errors_eigval, true_errors_eigval, alpha_array)
    plt.show()


if __name__ == "__main__":
    # A = np.diag((1, 2, 3))
    # A = np.arange(9).reshape((3,3))
    # x = np.array([1,2,3])
    # print(A)
    # print(x)
    # print(np.matmul(A, x))
    # print(x.T.dot(np.matmul(A, x)))
    # main_test()
    main()
