import math


def mean(values):
    return sum(values) / len(values)


def check_data(x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("X and Y must have the same length")

    if len(x_values) < 2:
        raise ValueError("At least 2 data pairs are required")

    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("X and Y must be finite numbers")


def compute_slope(x_values, y_values, x_mean, y_mean):
    xy_total = 0
    x_total = 0

    for i in range(len(x_values)):
        x_diff = x_values[i] - x_mean
        y_diff = y_values[i] - y_mean
        xy_total += x_diff * y_diff
        x_total += x_diff ** 2

    if x_total == 0:
        raise ValueError("All X values are the same")

    return xy_total / x_total


def compute_intercept(x_mean, y_mean, slope):
    return y_mean - slope * x_mean


def compute_pearson_r(x_values, y_values, x_mean, y_mean):
    xy_total = 0
    x_total = 0
    y_total = 0

    for i in range(len(x_values)):
        x_diff = x_values[i] - x_mean
        y_diff = y_values[i] - y_mean
        xy_total += x_diff * y_diff
        x_total += x_diff ** 2
        y_total += y_diff ** 2

    bottom = math.sqrt(x_total * y_total)

    if bottom == 0:
        return None

    return xy_total / bottom


def predict_y(x, coefficient, slope):
    return coefficient + slope * x


def compute_residuals(x_values, y_values, coefficient, slope):
    residuals = []

    for i in range(len(x_values)):
        predicted_y = predict_y(x_values[i], coefficient, slope)
        residuals.append(y_values[i] - predicted_y)

    return residuals


def compute_mse(residuals):
    total = 0

    for residual in residuals:
        total += residual ** 2

    return total / len(residuals)


def run_linear_regression(x_values, y_values):
    check_data(x_values, y_values)

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    try:
        slope = compute_slope(x_values, y_values, x_mean, y_mean)
        coefficient = compute_intercept(x_mean, y_mean, slope)
        pearson_r = compute_pearson_r(x_values, y_values, x_mean, y_mean)
        residuals = compute_residuals(x_values, y_values, coefficient, slope)
        mse = compute_mse(residuals)
    except OverflowError:
        raise ValueError("Calculated values are too big")

    for value in [coefficient, slope, mse]:
        if not math.isfinite(value):
            raise ValueError("Calculated values are too big")

    return {
        "coefficient": coefficient,
        "slope": slope,
        "pearson_r": pearson_r,
        "residuals": residuals,
        "mse": mse
    }


def check_number(test_name, expected, actual):
    if abs(expected - actual) < 0.000001:
        print("PASS:", test_name)
    else:
        print("FAIL:", test_name, "- expected", expected, "but got", actual)


def check_error(test_name, x_values, y_values):
    try:
        run_linear_regression(x_values, y_values)
        print("FAIL:", test_name, "- no error was raised")
    except ValueError as error:
        print("PASS:", test_name, "-", error)


if __name__ == "__main__":
    result = run_linear_regression([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
    check_number("Normal 1 slope", 0.6, result["slope"])
    check_number("Normal 1 intercept", 2.2, result["coefficient"])
    check_number("Normal 1 pearson r", math.sqrt(0.6), result["pearson_r"])
    check_number("Normal 1 mse", 0.48, result["mse"])

    result = run_linear_regression([1, 2, 3, 4], [8, 6, 4, 2])
    check_number("Normal 2 slope", -2, result["slope"])
    check_number("Normal 2 pearson r", -1, result["pearson_r"])

    result = run_linear_regression([1, 2, 3], [2, 4, 2])
    check_number("Normal 3 slope", 0, result["slope"])
    check_number("Normal 3 pearson r", 0, result["pearson_r"])

    result = run_linear_regression([1, 3], [1, 5])
    check_number("Boundary 2 points slope", 2, result["slope"])
    check_number("Boundary 2 points intercept", -1, result["coefficient"])

    result = run_linear_regression([1, 2, 3], [4, 4, 4])
    if result["pearson_r"] is None:
        print("PASS: Boundary all Y the same, pearson r is None")
    else:
        print("FAIL: Boundary all Y the same, pearson r should be None")

    check_error("Invalid all X the same", [2, 2, 2], [1, 2, 3])
    check_error("Invalid only 1 pair", [1], [1])
    check_error("Invalid different lengths", [1, 2, 3], [1, 2])
    check_error("Invalid NaN value", [1, 2, float("nan")], [1, 2, 3])
