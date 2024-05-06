from typing import Callable

import numpy as np
from enum import Enum


DEBUG = False


class RegressionType(Enum):
    Linear = 0,
    Quadratic = 1,
    Hyperbolic = 2,
    Exponential = 3,
    Power = 4,
    Logarithmic = 5,


# noinspection PyTypeChecker
def get_linear_regression(y: np.array, x: np.array) -> Callable[[float], float]:
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.matrix(np.array([[sum(x ** 2), sum(x)],
                            [sum(x),      len(y)]]))

    b = np.matrix([sum(x * y), sum(y)]).transpose()
    coefficients = np.asfarray(np.linalg.lstsq(a, b, -1)[0])

    if DEBUG:
        print(f"Coefficients of Linear regression are\n{coefficients}")

    return lambda v1: (coefficients[0] * v1 + coefficients[1])[0]


# noinspection PyTypeChecker
def get_quadratic_regression(y: np.array, x: np.array) -> Callable[[float], float]:
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.matrix(np.array([[sum(x ** 4), sum(x ** 3), sum(x ** 2)],
                            [sum(x ** 3), sum(x ** 2), sum(x)],
                            [sum(x ** 2), sum(x),      len(y)]]))

    b = np.matrix([sum(x ** 2 * y), sum(x * y), sum(y)]).transpose()
    coefficients = np.asfarray(np.linalg.lstsq(a, b, -1)[0])

    if DEBUG:
        print(f"Coefficients of Quadratic regression are\n{coefficients}")

    return lambda v1: (coefficients[0] * v1 ** 2 + coefficients[1] * v1 + coefficients[2])[0]


# noinspection PyTypeChecker
def get_hyperbolic_regression(y: np.array, x: np.array) -> Callable[[float], float]:
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    if 0 in y or 0 in x:
        raise ValueError("Zeros cannot appear in the data")

    a = np.matrix(np.array([[sum(1 / (x ** 2)), sum(1 / x)],
                            [sum(1 / x),        len(y)]]))

    b = np.matrix([sum(y / x), sum(y)]).transpose()
    coefficients = np.asfarray(np.linalg.lstsq(a, b, -1)[0])

    if DEBUG:
        print(f"Coefficients of Hyperbolic regression are\n{coefficients}")

    return lambda v1: (coefficients[0] / v1 + coefficients[1])[0]


# noinspection PyTypeChecker
def get_exponential_regression(y: np.array, x: np.array) -> Callable[[float], float]:
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.matrix(np.array([[sum(x ** 2), sum(x)],
                            [sum(x),      len(y)]]))

    b = np.matrix([sum(x * np.log(y)), sum(np.log(y))]).transpose()
    coefficients = np.asfarray(np.linalg.lstsq(a, b, -1)[0])

    if DEBUG:
        print(f"Coefficients of Exponential regression are\n{coefficients}")

    return lambda v1: (np.exp(coefficients[1]) * np.exp(coefficients[0] * v1))[0]


# noinspection PyTypeChecker
def get_power_regression(y: np.array, x: np.array) -> Callable[[float], float]:
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    if 0 in y or 0 in x:
        raise ValueError("Zeros cannot appear in the data")

    a = np.matrix(np.array([[len(y),         sum(np.log(x))],
                            [sum(np.log(x)), sum(np.log(x) ** 2)]]))

    b = np.matrix([sum(np.log(y)), sum(np.log(x) * np.log(y))]).transpose()
    coefficients = np.asfarray(np.linalg.lstsq(a, b, -1)[0])

    if DEBUG:
        print(f"Coefficients of Power regression are\n{coefficients}")

    return lambda v1: (np.exp(coefficients[0]) * v1**coefficients[1])[0]


# noinspection PyTypeChecker
def get_logarithmic_regression(y: np.array, x: np.array) -> Callable[[float], float]:
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    if 0 in y or 0 in x:
        raise ValueError("Zeros cannot appear in the data")

    a = np.matrix(np.array([[sum(np.log(x) ** 2), sum(np.log(x))],
                            [sum(np.log(x)),      len(y)]]))

    b = np.matrix([sum(y * np.log(x)), sum(y)]).transpose()
    coefficients = np.asfarray(np.linalg.lstsq(a, b, -1)[0])

    if DEBUG:
        print(f"Coefficients of Logarithmic regression are\n{coefficients}")

    return lambda v1: (coefficients[0] * np.log(v1) + coefficients[1])[0]


def get_regression_equation(regression_type: RegressionType) -> Callable[[np.array, np.array], Callable[[float], float]]:
    match regression_type:
        case RegressionType.Linear:
            return get_linear_regression

        case RegressionType.Quadratic:
            return get_quadratic_regression

        case RegressionType.Hyperbolic:
            return get_hyperbolic_regression

        case RegressionType.Exponential:
            return get_exponential_regression

        case RegressionType.Power:
            return get_power_regression

        case RegressionType.Logarithmic:
            return get_logarithmic_regression

        case _:
            raise ValueError(f"Invalid regression type: {regression_type}")


def fit_regression(y: list, x: list, regression_type: RegressionType = None) -> Callable[[float], float]:
    if regression_type is None:
        regression_type = get_best_regression_type(y, x)

    y = np.array(y)
    x = np.array(x)
    eq = get_regression_equation(regression_type)

    return eq(y, x)


def regression_error(y: list, x: list, regression_type: RegressionType) -> float:
    fitted_regression = fit_regression(y, x, regression_type)
    return sum(abs(y[i] - fitted_regression(i)) for i in range(len(y))) / len(y)


def all_regression_errors(y: list, x: list) -> dict[RegressionType, float]:
    result = {}
    for regression_type in RegressionType:
        result[regression_type] = regression_error(y, x, regression_type)

    return result


def get_best_regression_type(y: list, x: list, exclude=None) -> RegressionType:
    errors = all_regression_errors(y, x)

    if exclude is not None:
        for regression_type in exclude:
            errors.pop(regression_type)

    return min(errors, key=errors.get)


def fit_best_regression(y: list, x: list) -> Callable[[float], float]:
    return fit_regression(y, x, get_best_regression_type(y, x))

