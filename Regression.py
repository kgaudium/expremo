import numpy as np
from enum import Enum


class RegressionType(Enum):
    Linear = 0,
    Quadratic = 1,
    Hyperbolic = 2,
    Exponential = 3,
    Power = 4,
    Logarithmic = 5


def get_linear_regression(y: np.array, x: np.array):
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.array([[sum(x ** 2), sum(x)],
                  [sum(x),      len(y)]])

    b = np.array([sum(x * y), sum(y)])
    coefficients = np.linalg.lstsq(a, b, -1)[0]

    return lambda v1: coefficients[0] * v1 + coefficients[1]


def get_quadratic_regression(y: np.array, x: np.array):
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.array([[sum(x ** 4), sum(x ** 3), sum(x ** 2)],
                  [sum(x ** 3), sum(x ** 2), sum(x)],
                  [sum(x ** 2), sum(x),      len(y)]])

    b = np.array([sum(x ** 2 * y), sum(x * y), sum(y)])
    coefficients = np.linalg.lstsq(a, b, -1)[0]

    return lambda v1: coefficients[0] * v1 ** 2 + coefficients[1] * v1 + coefficients[2]


def get_hyperbolic_regression(y: np.array, x: np.array):
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.array([[sum(1 / (x ** 2)), sum(1 / x)],
                  [sum(1 / x),        len(y)]])

    b = np.array([sum(y / x), sum(y)])
    coefficients = np.linalg.lstsq(a, b, -1)[0]

    return lambda v1: coefficients[0] / v1 + coefficients[1]


def get_exponential_regression(y: np.array, x: np.array):
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.array([[sum(x ** 2), sum(x)],
                  [sum(x),      len(y)]])

    b = np.array([sum(x * np.log(y)), sum(np.log(y))])
    coefficients = np.linalg.lstsq(a, b, -1)[0]

    return lambda v1: np.exp(coefficients[1]) * np.exp(coefficients[0] * v1)


def get_power_regression(y: np.array, x: np.array):
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.array([[len(y),         sum(np.log(x))],
                  [sum(np.log(x)), sum(np.log(x) ** 2)]])

    b = np.array([sum(np.log(y)), sum(np.log(x) * np.log(y))])
    coefficients = np.linalg.lstsq(a, b, -1)[0]

    return lambda v1: np.exp(coefficients[0]) * v1**coefficients[1]


def get_logarithmic_regression(y: np.array, x: np.array):
    if len(y) != len(x):
        raise ValueError(
            f"All inputs must have the same length. Got: {", ".join(map(lambda var: str(len(var)), [y, x]))}")

    a = np.array([[sum(np.log(x) ** 2), sum(np.log(x))],
                  [sum(np.log(x)),      len(y)]])

    b = np.array([sum(y * np.log(x)), sum(y)])
    coefficients = np.linalg.lstsq(a, b, -1)[0]

    return lambda v1: coefficients[0] * np.log(v1) + coefficients[1]


def get_regression_equation(regression_type: RegressionType):
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


def regression(y: list, x: list, regression_type: RegressionType):
    y = np.array(y)
    x = np.array(x)
    eq = get_regression_equation(regression_type)

    return eq(y, x)


if __name__ == "__main__":
    fnc = regression([1, 4, 7], [1, 3, 5], RegressionType.Linear)  # y(x)

    print(fnc(10))