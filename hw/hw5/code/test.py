import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

_beta = np.array([9, 1, 1, 1, 1])

"""
Returns the tuple train_X,train_y,valid_X,valid_y containing
the training and validation sets.
"""


def get_data_2():
    global _beta
    np.random.seed(1234)
    num_train, num_valid = 50, 20
    num_train, num_valid = 2, 2
    n = num_train + num_valid
    X = np.zeros((n, 1))
    X[:, 0] = 1
    y = np.dot(X, _beta)
    return X[:num_train, :], \
           y[:num_train], \
           X[num_train:, :], \
           y[num_train:]

def get_data():
        global _beta
        np.random.seed(1234)
        num_train, num_valid = 50, 20
        #num_train, num_valid = 2, 2
        n = num_train + num_valid
        X = np.zeros((n, 5))
        X[:, 0] = 1
        X[:, 1] = X[:, 0]  + np.random.randn(n)
        for i in range(2, 5):
            X[:, i] = X[:, i - 1] + np.random.randn(n) * 0.01
        w = np.random.randn(n)
        y = np.dot(X, _beta) + w
        return X[:num_train, :], \
               y[:num_train], \
               X[num_train:, :], \
               y[num_train:]

def compute_loss_square_with_true_betas(X, y, beta):
    loss = (np.dot(X, beta) - y) ** 2
    loss = np.sum(loss, axis=0)
    return loss
    # return ((np.matmul(X, beta) - y) ** 2).sum()


def compute_square_loss(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    return np.sum(loss, axis=0)


def main():
    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = np.dot(X, np.array([1, 2]))
    # reg = LinearRegression().fit(X, y)
    # print(reg.coef_)
    # y_pred = reg.predict(X)
    # print(compute_square_loss(y, y_pred))
    # print(np.dot(X, reg.coef_), y)
    # print(compute_loss_square_with_true_betas(X, y, reg.coef_))
    train_X, train_y, valid_X, valid_y = get_data()


    print("********* Linear Regression *********")
    linear_regression = LinearRegression(fit_intercept=False)
    linear_regression.fit(train_X, train_y)
    train_y_predictions = linear_regression.predict(train_X)
    valid_y_predictions = linear_regression.predict(valid_X)
    #
    # The betas
    linear_reg_beta = linear_regression.coef_
    print("True betas:{}\nEstimated linear regression betas:{}\n".format(_beta, linear_reg_beta))

    print("Training loss with true beta:{:.3f}\nValidation loss with true beta:{:.3f}"
          .format(compute_loss_square_with_true_betas(train_X, train_y, _beta),
                  compute_loss_square_with_true_betas(valid_X, valid_y, _beta)))
    print("Training square loss:{:.3f}\nValidation square loss:{:.3f}"
          .format(compute_square_loss(train_y, train_y_predictions),
                  compute_square_loss(valid_y, valid_y_predictions)))
    X = np.vstack([train_X, valid_X])
    y = np.hstack([train_y, valid_y])
    #print("X:{}".format(X))
    #print("y:{}".format(y))
    print("Rank:{:d}".format(np.linalg.matrix_rank(X)))

    print("********* Ridge Regression *********")
    ridge_regression = Ridge(alpha=0.5,fit_intercept=False)
    ridge_regression.fit(train_X, train_y)
    train_y_predictions = ridge_regression.predict(train_X)
    valid_y_predictions = ridge_regression.predict(valid_X)

    ridge_beta = ridge_regression.coef_
    print("True beta:{}\nRidge betas:{}\n".format(_beta, ridge_beta))

    print("Training loss with true beta:{:.3f}\nValidation loss with true beta:{:.3f}"
          .format(compute_loss_square_with_true_betas(train_X, train_y, _beta),
                  compute_loss_square_with_true_betas(valid_X, valid_y, _beta)))
    print("Training square loss:{:.3f}\nValidation square loss:{:.3f}"
          .format(compute_square_loss(train_y, train_y_predictions),
                  compute_square_loss(valid_y, valid_y_predictions)))

    x, res, rank, s = np.linalg.lstsq(train_X, train_y, rcond=-1)
    print(rank, s)
    print(x)
    print(res)


if __name__ == "__main__":
    main()