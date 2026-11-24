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

        if y <= 0:
            raise ValueError("Y values must be greater than zero")


def compute_slope(x_values, log_y_values, x_mean, log_y_mean):
    xy_total = 0
    x_total = 0

    for i in range(len(x_values)):
        x_diff = x_values[i] - x_mean
        y_diff = log_y_values[i] - log_y_mean
        xy_total += x_diff * y_diff
        x_total += x_diff ** 2

    if x_total == 0:
        raise ValueError("All X values are the same")

    return xy_total / x_total


def compute_coefficient(x_mean, log_y_mean, slope):
    intercept = log_y_mean - slope * x_mean
    return math.exp(intercept)


def compute_pearson_r(x_values, log_y_values, x_mean, log_y_mean):
    xy_total = 0
    x_total = 0
    y_total = 0

    for i in range(len(x_values)):
        x_diff = x_values[i] - x_mean
        y_diff = log_y_values[i] - log_y_mean
        xy_total += x_diff * y_diff
        x_total += x_diff ** 2
        y_total += y_diff ** 2

    bottom = math.sqrt(x_total * y_total)

    if bottom == 0:
        return None

    return xy_total / bottom


def predict_y(x, coefficient, slope):
    return coefficient * math.exp(slope * x)


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


def run_exponential_regression(x_values, y_values):
    check_data(x_values, y_values)

    log_y_values = []
    for y in y_values:
        log_y_values.append(math.log(y))

    x_mean = mean(x_values)
    log_y_mean = mean(log_y_values)

    try:
        slope = compute_slope(x_values, log_y_values, x_mean, log_y_mean)
        coefficient = compute_coefficient(x_mean, log_y_mean, slope)
        pearson_r = compute_pearson_r(x_values, log_y_values, x_mean, log_y_mean)
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
