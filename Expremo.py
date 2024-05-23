from typing import Callable

import Regression
from matplotlib import pyplot as plt
from enum import Enum


class TrendType(Enum):
    Downward = -1
    Stable = 0
    Upward = 1


class Expremo:
    def __init__(self, price_data=None, time_data=None, regression_type=None):
        self.time_data = time_data
        self.price_data = price_data
        self.regression_type = regression_type

        self.train_data = price_data, time_data
        self.regression_func = None
        self.train_range = None
        self.excluded_types = []
        self.trend_threshold_percentage = 0.05

    def set_time_data(self, time_data: list) -> None:
        self.time_data = time_data

    def set_price_data(self, price_data: list) -> None:
        self.price_data = price_data

    def set_regression_type(self, regression_type: Regression.RegressionType) -> None:
        self.regression_type = regression_type
        self.update_regression_func()

    def set_train_data(self, train_data: tuple[list, list]) -> None:
        self.train_data = train_data

    def add_excluded_type(self, excluded_type: Regression.RegressionType) -> None:
        self.excluded_types += [excluded_type]

    def clear_excluded_types(self) -> None:
        self.excluded_types = []

    def update_regression_func(self) -> None:
        self.regression_func = Regression.fit_regression(*self.train_data, self.regression_type)

    def choose_best_regression_type(self) -> None:
        self.set_regression_type(Regression.get_best_regression_type(*self.train_data, self.excluded_types))

    def set_train_range(self, train_range: tuple[float, float]) -> None:
        self.train_range = train_range

    def predict(self, start_point: float, train_distance: float, prediction_distance: float) -> float:
        self.set_train_range((start_point, start_point + train_distance))

        if self.train_range[1] == 521:
            pass

        self.set_train_data(self.get_data_slice(*self.train_range))
        self.choose_best_regression_type()
        self.update_regression_func()

        func = self.get_regression_func()
        predict_point = start_point + train_distance + prediction_distance

        return func(predict_point)

    def predict_trend(self, start_point: float, train_distance: float, prediction_distance: float) -> TrendType:
        predicted_point = self.predict(start_point, train_distance, prediction_distance)
        at_start_point = self.get_regression_func()(start_point)
        predicted_trend = self.calculate_trend(at_start_point, predicted_point)
        return predicted_trend

    def calculate_trend(self, start: float, end: float) -> TrendType:
        diff_perc = abs(1 - end / start)
        if diff_perc <= self.trend_threshold_percentage:
            return TrendType.Stable

        if start - end > 0:
            return TrendType.Downward
        else:
            return TrendType.Upward

    def get_regression_func(self) -> Callable[[float], float]:
        if self.regression_func is None:
            self.update_regression_func()

        return self.regression_func

    def get_data_slice(self, start: float, end: float) -> tuple[list, list]:
        return self.price_data[start:end], self.time_data[start:end]


def plot_prediction(expremo: Expremo, prediction_distance: float, plot_scatter=True) -> None:
    plot_range = expremo.train_range[0], expremo.train_range[1]

    # Plots original data
    plt.plot(expremo.time_data[plot_range[0] : plot_range[1] + prediction_distance],
             expremo.price_data[plot_range[0] : plot_range[1] + prediction_distance],
             color='blue',
             linewidth=1)

    # Extends time_data if needed
    prediction_time_data = expremo.time_data.copy()
    if plot_range[1] > len(expremo.time_data):
        prediction_time_data += list(range(prediction_time_data[-1],
                                           prediction_time_data[-1] + plot_range[1] - len(expremo.time_data)))

    # trend = expremo.predict_trend(expremo.train_range[0], expremo.train_range[1], prediction_distance)
    trend = expremo.calculate_trend(expremo.get_regression_func()(plot_range[0]), expremo.get_regression_func()(plot_range[1]+prediction_distance))
    pred_plot_color = None
    match trend:
        case TrendType.Upward:
            pred_plot_color = 'green'
        case TrendType.Downward:
            pred_plot_color = 'red'
        case TrendType.Stable:
            pred_plot_color = 'gray'
        case _:
            raise ValueError(f'Trend type {trend} not supported')

    # Plots Regression func
    plt.plot(prediction_time_data[slice(*plot_range)],
             [expremo.get_regression_func()(i) for i in range(*plot_range)],
             color=pred_plot_color)

    # Plots Predicted Point
    if plot_scatter:
        prediction_point = plot_range[1] + prediction_distance
        plt.scatter(prediction_point,
                    expremo.get_regression_func()(prediction_point),
                    color='orange')


def calculate_absolute_prediction_error(expremo: Expremo, prediction_distance: float) -> float:
    prediction_point = expremo.train_range[1] + prediction_distance

    prediction = expremo.get_regression_func()(prediction_point)
    try:
        real_value = expremo.price_data[prediction_point]
    except IndexError:
        pass


    return abs(prediction - real_value)


def calculate_relative_prediction_error(expremo: Expremo, prediction_distance: float) -> float:
    prediction_point = expremo.train_range[1] + prediction_distance
    absolute_error = calculate_absolute_prediction_error(expremo, prediction_distance)

    return abs(absolute_error / expremo.price_data[prediction_point])


def is_correct_trend(expremo: Expremo, prediction_distance: float) -> bool:
    # Считает с конца выборки обучения, потому что покупать мы будем именно в этот момент
    prediction_point = expremo.train_range[1] + prediction_distance

    predicted_trend = expremo.predict_trend(expremo.train_range[0],
                                            expremo.train_range[1]-expremo.train_range[0],
                                            prediction_distance)

    real_trend = expremo.calculate_trend(expremo.price_data[expremo.train_range[1]], expremo.price_data[prediction_point])

    return predicted_trend == real_trend


def calculate_profit(expremo: Expremo, prediction_distance: float) -> float:
    '''Считает полученую выгоду с учётом того, что акцию купили в конце обучающего диапазона, а продали на расстояние предсказания от него.'''
    # Считает с конца выборки обучения, потому что покупать мы будем именно в этот момент
    prediction_point = expremo.train_range[1] + prediction_distance
    # prediction = expremo.get_regression_func()(prediction_point)
    return expremo.price_data[prediction_point] - expremo.price_data[expremo.train_range[1]]
