import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def get_model_data(num_samples: int, T: int) -> np.array:
    t = np.arange(num_samples)
    X = np.zeros((t.shape[0], 4))
    X[:, 0] = np.ones_like(t)
    X[:, 1] = t
    X[:, 2] = np.cos((2 * np.pi * t) / T)
    X[:, 3] = np.sin((2 * np.pi) / T * t)
    return X


def load_data():
    df = pd.read_csv('t_data.csv')
    print("Features=", df.columns)
    print("Num Years=", len(df) // 12, "Num Months=", len(df))
    df['month'] = df.index
    return df[0:150 * 12], df[150 * 12:]


def predict_using_all_training_data(data, T):
    model = LinearRegression(fit_intercept=True)
    n_samples = data.shape[0]
    X = get_model_data(n_samples, T)
    #U, S, V = np.linalg.svd(X)
    #print(S)
    model.fit(X, data)
    predictions = model.predict(X)
    #print(model.coef_.shape)
    training_error = mean_squared_error(predictions, data, squared=False)
    return training_error

def determine_best_T_using_all_training_data(train_max_temp):
    training_errors = list()
    for T in range(1, 21):
        training_error = predict_using_all_training_data(train_max_temp, T)
        training_errors.append(training_error)

    training_errors = np.array(training_errors)
    print("Minimum training error at T={:d}".format(np.argmin(training_errors)))
    plt.figure(figsize=(9, 6))
    t = np.arange(1, 21)
    plt.plot(t, training_errors, "-o", lw=3, color='purple', label="Training error")
    plt.xlabel("Period T", fontsize=15, labelpad=10)
    plt.ylabel("RMSE", fontsize=15, labelpad=10)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xmin=0)
    plt.xticks(np.arange(0, 21, step=2))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig("train_errors.pdf")

def predict_using_training_validation_set(data, T):
    train, val = train_test_split(data, test_size=0.2, random_state=1234)
    model = LinearRegression(fit_intercept=True)
    n_train = train.shape[0]
    n_val = val.shape[0]
    X = get_model_data(n_train + n_val, T)
    # U, S, V = np.linalg.svd(X)
    # print(S)
    X_train = X[:n_train, :]
    model.fit(X_train, train)
    train_predictions = model.predict(X_train)
    training_error = mean_squared_error(train_predictions, train, squared=False)
    X_val = X[n_train:, :]
    val_predictions = model.predict(X_val)
    validation_error = mean_squared_error(val_predictions, val, squared=False)

    return training_error, validation_error


def determine_best_T_using_training_validation(train_max_temp):
    training_errors = list()
    validation_errors = list()
    for T in range(1, 21):
        training_error, validation_error = predict_using_training_validation_set(train_max_temp, T)
        training_errors.append(training_error)
        validation_errors.append(validation_error)

    training_errors = np.array(training_errors)
    validation_errors = np.array(validation_errors)
    print("Minimum training error at T={:d}, Minimum validation error at T:{:d}".format(np.argmin(training_errors),
                                                                                 np.argmin(validation_errors)))
    plt.figure(figsize=(9, 6))
    t = np.arange(1, 21)
    plt.plot(t, training_errors, "-o", lw=3, color='purple', label="Training error")
    plt.plot(t, validation_errors, "-o", lw=3, color='crimson', label="Validation error")
    plt.xlabel("Period T", fontsize=15, labelpad=10)
    plt.ylabel("RMSE", fontsize=15, labelpad=10)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xmin=0)
    plt.xticks(np.arange(0, 21, step=2))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig("train_val_errors.pdf")
    plt.show()


def plot_fit(X, y, b, name):
    plt.plot(X.t, y, label='Actual')
    plt.plot(X.t, X @ b, label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Max Temperature (C)')
    plt.title('Max Temperature - ' + name)
    plt.legend()
    plt.savefig('%s_fit.pdf' % name, bbox_inches='tight')
    plt.close()


def main():
    np.random.seed(1234)
    train, test = load_data()
    train_max_temp = np.array(train["max_temp_C"])[..., np.newaxis]
    test_max_temp = np.array(test["max_temp_C"])[..., np.newaxis]

    #determine_best_T_using_all_training_data(train_max_temp)
    determine_best_T_using_training_validation(train_max_temp)


if __name__ == "__main__":
    main()
