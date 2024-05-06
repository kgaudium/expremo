import Regression
from matplotlib import pyplot as plt


# TODO: set typing
class Expremo:
    def __init__(self, price_data=None, time_data=None, regression_type=None):
        self.time_data = time_data
        self.price_data = price_data
        self.regression_type = regression_type

        self.train_data = price_data, time_data
        self.regression_func = None
        self.train_range = None
        self.excluded_types = []

    def set_time_data(self, time_data):
        self.time_data = time_data

    def set_price_data(self, price_data):
        self.price_data = price_data

    def set_regression_type(self, regression_type):
        self.regression_type = regression_type
        self.update_regression_func()

    def set_train_data(self, train_data):
        self.train_data = train_data
        self.set_best_regression_type()

    def add_excluded_type(self, excluded_type):
        self.excluded_types += [excluded_type]

    def clear_excluded_types(self):
        self.excluded_types = []

    def update_regression_func(self):
        self.regression_func = Regression.fit_regression(*self.train_data, self.regression_type)

    def set_best_regression_type(self):
        self.regression_type = Regression.get_best_regression_type(*self.train_data, self.excluded_types)
        self.update_regression_func()

    def predict(self, start_point, train_distance, prediction_distance):
        self.train_range = (start_point, train_distance)

        self.set_train_data(self.get_data_slice(*self.train_range))
        func = self.get_regression_func()
        predict_point = start_point + train_distance + prediction_distance

        return func(predict_point)

    def get_regression_func(self):
        if self.regression_func is None:
            self.update_regression_func()

        return self.regression_func

    def get_data_slice(self, start, end):
        return self.price_data[start:end], self.time_data[start:end]


# считать ошибку/точность предсказания
def plot_prediction(expremo, prediction_distance):
    plot_range = expremo.train_range[0], expremo.train_range[1]

    plt.plot(expremo.time_data[plot_range[0] : plot_range[1] + prediction_distance],
             expremo.price_data[plot_range[0] : plot_range[1] + prediction_distance],
             color='cyan')

    prediction_time_data = expremo.time_data.copy()
    if plot_range[1] > len(expremo.time_data):
        prediction_time_data += list(range(prediction_time_data[-1],
                                           prediction_time_data[-1] + plot_range[1] - len(expremo.time_data)))

    plt.plot(prediction_time_data[slice(*plot_range)],
             [expremo.get_regression_func()(i) for i in range(*plot_range)],
             color='red')

    prediction_point = plot_range[1] + prediction_distance
    plt.scatter(prediction_point,
                expremo.get_regression_func()(prediction_point),
                color='orange')
