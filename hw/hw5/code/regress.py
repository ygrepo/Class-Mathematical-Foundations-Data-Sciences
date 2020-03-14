import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

_beta = np.array([9, 1, 1, 1, 1])

"""
Returns the tuple train_X,train_y,valid_X,valid_y containing
the training and validation sets.
"""


def get_data():
    global _beta
    np.random.seed(1234)
    num_train, num_valid = 50, 20
    n = num_train + num_valid
    X = np.zeros((n, 5))
    X[:, 0] = 1
    X[:, 1] = X[:, 0] + np.random.randn(n)
    for i in range(2, 5):
        X[:, i] = X[:, i - 1] + np.random.randn(n) * 0.01
    w = np.random.randn(n)
    y = np.dot(X, _beta) + w
    return X[:num_train, :], \
           y[:num_train], \
           X[num_train:, :], \
           y[num_train:], \
           w[:num_train], \
           w[num_train:]


def compute_loss_square_with_true_betas(X, y, beta):
    loss = (np.dot(X, beta) - y) ** 2
    loss = np.sum(loss, axis=0)
    return loss
    # return ((np.matmul(X, beta) - y) ** 2).sum()


def compute_square_loss(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    return np.sum(loss, axis=0)


def main():
    train_X, train_y, valid_X, valid_y, train_w, val_w = get_data()

    print("********* Linear Regression *********")
    linear_regression = LinearRegression(fit_intercept=False)
    linear_regression.fit(train_X, train_y)
    train_y_predictions = linear_regression.predict(train_X)
    valid_y_predictions = linear_regression.predict(valid_X)

    # The betas
    linear_reg_beta = linear_regression.coef_
    print("True betas:{}\nEstimated linear regression betas:{}\n".format(_beta, linear_reg_beta))

    print("Training loss with true beta:{:.3f}\nValidation loss with true beta:{:.3f}"
          .format(compute_loss_square_with_true_betas(train_X, train_y, _beta),
                  compute_loss_square_with_true_betas(valid_X, valid_y, _beta)))
    print("Training square loss:{:.3f}\nValidation square loss:{:.3f}"
          .format(compute_square_loss(train_y, train_y_predictions),
                  compute_square_loss(valid_y, valid_y_predictions)))

    # The mean squared error
    print("Linear regression mean squared error: {:.2f}, {:.2f}\n"
          .format(mean_squared_error(train_y, train_y_predictions),
                  mean_squared_error(valid_y, valid_y_predictions)))

    print("********* Linear Regression Error Analysis *********")
    X = np.vstack([train_X, valid_X])
    print("Rank:{:d}".format(np.linalg.matrix_rank(X)))
    U, S, V = np.linalg.svd(X, full_matrices=False)
    X_SVD = U @ np.diag(S) @ V
    print("Is X close to X_SVD?", np.isclose(X, X_SVD).all())
    print("Singular values:{}".format(S))
    w = np.hstack([train_w, val_w])
    Inv_S = np.linalg.inv(np.diag(S))
    print("True betas:{}".format(_beta))
    print("Computed betas:{}".format(linear_reg_beta))
    beta_OLS_beta_true = (U @ Inv_S @ V).T @ w
    print("Difference:{}".format(beta_OLS_beta_true))
    mean_training_error = 1 + 5 / 70
    variance_training_error = (70 - 5) / (70. ** 2)
    variance_training_error *= 2
    print("Mean of the average training error:{}\nVariance:{}".format(mean_training_error, variance_training_error))

    valid_x_cov = np.cov(valid_X)
    U = U[:20, :5]
    test_mean_square_error = U.T @ valid_x_cov @ U
    test_mean_square_error = test_mean_square_error @ Inv_S
    test_mean_square_error = np.sum(test_mean_square_error, axis=1)
    print("Test mean square error:{}\n".format(test_mean_square_error))

    print("********* Ridge Regression *********")
    ridge_regression = Ridge(alpha=0.5,fit_intercept=False)
    ridge_regression.fit(train_X, train_y)
    train_y_predictions = ridge_regression.predict(train_X)
    valid_y_predictions = ridge_regression.predict(valid_X)

    # The coefficients
    ridge_beta = ridge_regression.coef_
    print("True beta:{}\nRidge betas:{}\n".format(_beta, ridge_beta))

    print("Training loss with true beta:{:.3f}\nValidation loss with true beta:{:.3f}"
          .format(compute_loss_square_with_true_betas(train_X, train_y, _beta),
                  compute_loss_square_with_true_betas(valid_X, valid_y, _beta)))

    print("Training square loss:{:.3f}\nValidation square loss:{:.3f}"
          .format(compute_square_loss(train_y, train_y_predictions),
                  compute_square_loss(valid_y, valid_y_predictions)))

    # The mean squared error
    print('Mean squared error: {:.2f}, {:.2f}'
          .format(mean_squared_error(train_y, train_y_predictions),
                  mean_squared_error(valid_y, valid_y_predictions)))

    U, S, V = np.linalg.svd(train_X, full_matrices=False)
    X_SVD = U @ np.diag(S) @ V
    print("Is X close to X_SVD?", np.isclose(train_X, X_SVD).all())
    print("Singular values:{}".format(S))
    # print(S * np.ones(5))
    # Plot outputs
    # plt.scatter(train_X, train_y, color='black')
    # plt.plot(train_X, train_y_predictions, color='blue', linewidth=3)


if __name__ == "__main__":
    main()
