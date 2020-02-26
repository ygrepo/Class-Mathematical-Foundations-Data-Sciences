import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def compute_error(less_a, less_a_pred, greater_a, greater_a_pred):
    less_a_error = mean_squared_error(less_a, less_a_pred)
    # less_a_train_sample_weights = np.array([1. / rain_train_less_a_pred.shape[0]])
    # less_a_train_sample_weights = np.tile(less_a_train_sample_weights, rain_train_less_a_pred.shape[0])

    greater_a_error = mean_squared_error(greater_a, greater_a_pred)

    n_samples_less_a = less_a.shape[0]
    n_samples_greater_a = greater_a.shape[0]
    error = n_samples_less_a * less_a_error + n_samples_greater_a * greater_a_error
    error = error / (n_samples_less_a + n_samples_greater_a)
    print("{},{}, {}".format(less_a_error, greater_a_error, error))
    return error

    # greater_a_train_sample_weights = np.array([1. / rain_train_greater_a_pred.shape[0]])
    # greater_a_train_sample_weights = np.tile(greater_a_train_sample_weights, rain_train_greater_a_pred.shape[0])
    #
    # stacked_rain_train = np.hstack((rain_train_less_a, rain_train_greater_a))
    # stacked_rain_train_pred = np.hstack((rain_train_less_a_pred, rain_train_greater_a_pred))
    # stacked_train_sample_weights = np.hstack((less_a_train_sample_weights, greater_a_train_sample_weights))
    # train_error = mean_squared_error(stacked_rain_train, stacked_rain_train_pred, stacked_train_sample_weights)


def split_and_plot(a, max_temp_train, rain_train, max_temp_val, rain_val, grid):
    grid_less_a = grid[grid < a].reshape(-1, 1)
    grid_greater_a = grid[grid >= a].reshape(-1, 1)

    # Split the training set into two data sets less or greater then temperature a
    max_temp_train_less_a = max_temp_train[np.where(max_temp_train < a)].reshape(-1, 1)
    max_temp_train_greater_a = max_temp_train[np.where(max_temp_train >= a)].reshape(-1, 1)

    # Do the same for the rainfall
    rain_train_less_a = rain_train[np.where(max_temp_train < a)]
    rain_train_greater_a = rain_train[np.where(max_temp_train >= a)]

    # fill in code to assign values for the variables below.
    # reg_1 = SGDRegressor()
    reg_1 = LinearRegression()
    # We fit a linear estimator for temperatures less than a
    reg_1.fit(max_temp_train_less_a, rain_train_less_a)

    # prediction values on relevant points on grid using linear fit with points max_temp < a
    linear_fit_rain_less_a = reg_1.predict(grid_less_a)

    # reg_2 = SGDRegressor()
    reg_2 = LinearRegression()

    reg_2.fit(max_temp_train_greater_a, rain_train_greater_a)
    # prediction values on relevant points on grid using linear fit with points max_temp > a
    linear_fit_rain_greater_a = reg_2.predict(grid_greater_a)

    plt.figure(figsize=(9, 6))
    plt.scatter(max_temp_val, rain_val, s=5, c="dodgerblue", marker='o', edgecolor="skyblue")

    plt.plot(grid_less_a, linear_fit_rain_less_a, '-o', lw=3, color='purple', label="<a")
    plt.plot(grid_greater_a, linear_fit_rain_greater_a, '-o', lw=3, color='crimson', label=">a")
    plt.axvline(a, ymin=0, ymax=np.max(max_temp_train), color='k', linestyle='dotted')
    plt.ylabel("Rain", fontsize=15, labelpad=10)
    plt.xlabel("Maximum temperature", fontsize=15, labelpad=10)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    # use a relevant error metric below. make sure you normalize correctly to take into account number of datapoints.
    rain_train_less_a_pred = reg_1.predict(max_temp_train_less_a)
    rain_train_greater_a_pred = reg_2.predict(max_temp_train_greater_a)
    train_error = compute_error(rain_train_less_a, rain_train_less_a_pred, rain_train_greater_a,
                                rain_train_greater_a_pred)

    # And we split the same way the validation dataset related to the temperatures
    max_temp_val_less_a = max_temp_val[np.where(max_temp_val < a)].reshape(-1, 1)
    rain_val_less_a = rain_val[np.where(max_temp_val < a)]
    rain_val_less_a_pred = reg_1.predict(max_temp_val_less_a)

    max_temp_val_greater_a = max_temp_val[np.where(max_temp_val >= a)].reshape(-1, 1)
    rain_val_greater_a = rain_val[np.where(max_temp_val >= a)]
    rain_val_greater_a_pred = reg_2.predict(max_temp_val_greater_a)
    val_error = compute_error(rain_val_less_a, rain_val_less_a_pred, rain_val_greater_a, rain_val_greater_a_pred)

    # val_error =  # error of your fit on validation data points
    # train_error =  # error of your fit on your train data points
    plt.title(
        'a = ' + str(a) + ' val error: ' + str(round(val_error, 3)) + ' train error: ' + str(round(train_error, 3)),
        fontsize=18)
    plt.savefig('a=' + str(a) + '.pdf')
    # plt.show()


def main():
    dataset = np.loadtxt("oxford_temperatures.txt")

    max_temp = np.array(dataset[:, 2])
    rain = np.array(dataset[:, 5])

    max_temp_train, max_temp_test, rain_train, rain_test = train_test_split(max_temp, rain, test_size=0.30,
                                                                            random_state=42)
    max_temp_val, max_temp_test, rain_val, rain_test = train_test_split(max_temp_test, rain_test, test_size=0.50,
                                                                        random_state=42)

    max_temp_train = max_temp_train.reshape(-1, 1)
    max_temp_test = max_temp_test.reshape(-1, 1)
    rain_train = rain_train.reshape(-1, 1)

    width_bin = 0.5
    max_val = np.max(max_temp)
    grid = np.arange(0, max_val + 1, width_bin)
    grid = grid.reshape(-1, 1)

    # for a in [4]:
    for a in np.arange(4, 25, 4):
        split_and_plot(a, max_temp_train, rain_train, max_temp_val, rain_val, grid)

    ### selecting the best model according to val accuracy
    a = 8  # best value of a from grid search above

    grid_less_a = grid[grid < a].reshape(-1, 1)
    grid_greater_a = grid[grid >= a].reshape(-1, 1)

    # regular linear regression on the entire train dataset
    reg_model = LinearRegression()
    reg_model.fit(max_temp_train, rain_train)
    linear_fit_rain = reg_model.predict(grid)

    best_model = LinearRegression()
    max_temp_less_a = max_temp_train[np.where(max_temp_train < a)].reshape(-1, 1)
    rain_less_a = rain_train[np.where(max_temp_train < a)]
    # fit from your best model/best a
    best_model.fit(max_temp_less_a, rain_less_a)
    linear_fit_rain_less_a = best_model.predict(grid_less_a)

    best_model = LinearRegression()
    max_temp_greater_a = max_temp_train[np.where(max_temp_train >= a)].reshape(-1, 1)
    rain_greater_a = rain_train[np.where(max_temp_train >= a)]
    #  fit from your best model/best a
    best_model.fit(max_temp_greater_a, rain_greater_a)
    linear_fit_rain_greater_a = best_model.predict(grid_greater_a)

    plt.figure(figsize=(9, 6))
    plt.scatter(max_temp_test, rain_test, s=5, c="dodgerblue", marker='o', edgecolor="skyblue")

    plt.plot(grid, linear_fit_rain, '--', lw=3, color='purple', label="Linear regression")
    plt.plot(grid_less_a, linear_fit_rain_less_a, '-o', lw=3, color='crimson')
    plt.plot(grid_greater_a, linear_fit_rain_greater_a, '-o', lw=3, color='crimson', label="a=" + str(a))
    plt.axvline(a, ymin=0, ymax=np.max(max_temp), color='k', linestyle='dotted')
    plt.ylabel("Rain", fontsize=15, labelpad=10)
    plt.xlabel("Maximum temperature", fontsize=15, labelpad=10)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    a_error = compute_error(grid_less_a, linear_fit_rain_less_a, grid_greater_a, linear_fit_rain_greater_a)
    lr_error = mean_squared_error(grid, linear_fit_rain)

    # a_error =  # error of your best model
    # lr_error =  # error of regular linear regression

    plt.title('a=' + str(a) + '_error: ' + str(round(a_error, 3)) + ' LR_error: ' + str(round(lr_error, 3)),
              fontsize=18)
    plt.savefig('test_comparison.pdf')


if __name__ == "__main__":
    main()
