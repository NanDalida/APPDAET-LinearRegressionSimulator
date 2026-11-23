import math

def mean(values):
    return sum(values) / len(values)

def compute_slope(log_xs, ys, x_mean, y_mean):
    numerator = 0
    denominator = 0

    for i in range(len(log_xs)):
        x_diff = log_xs[i] - x_mean
        y_diff = ys[i] - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff ** 2

    if denominator == 0:
        raise ValueError("All X values are the same")

    return numerator / denominator

def compute_coefficient(x_mean, y_mean, slope):
    return y_mean - slope * x_mean

def compute_pearson_r(log_xs, ys, x_mean, y_mean):
    numerator = 0
    x_total = 0
    y_total = 0

    for i in range(len(log_xs)):
        x_diff = log_xs[i] - x_mean
        y_diff = ys[i] - y_mean
        numerator += x_diff * y_diff
        x_total += x_diff ** 2
        y_total += y_diff ** 2

    denominator = math.sqrt(x_total * y_total)
    if denominator == 0:
        return None

    return numerator / denominator

def predict_y(x, coefficient, slope):
    return coefficient + slope * math.log(x)

def compute_residuals(xs, ys, coefficient, slope):
    residuals = []

    for i in range(len(xs)):
        predicted = predict_y(xs[i], coefficient, slope)
        residuals.append(ys[i] - predicted)

    return residuals

def compute_mse(residuals):
    total = 0

    for residual in residuals:
        total += residual ** 2

    return total / len(residuals)

def run_logarithmic_regression(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("X and Y must have the same length")

    if len(xs) < 2:
        raise ValueError("At least 2 data pairs are required")

    log_xs = []

    for i in range(len(xs)):
        if not math.isfinite(xs[i]) or not math.isfinite(ys[i]):
            raise ValueError("X and Y must be finite numbers")

        if xs[i] <= 0:
            raise ValueError("X values must be greater than zero")

        log_xs.append(math.log(xs[i]))

    x_mean = mean(log_xs)
    y_mean = mean(ys)
    slope = compute_slope(log_xs, ys, x_mean, y_mean)

    try:
        coefficient = compute_coefficient(x_mean, y_mean, slope)
        pearson_r = compute_pearson_r(log_xs, ys, x_mean, y_mean)
        residuals = compute_residuals(xs, ys, coefficient, slope)
        mse = compute_mse(residuals)

    except OverflowError:
        raise ValueError("Calculated values are too big")

    if not math.isfinite(coefficient) or not math.isfinite(slope) or not math.isfinite(mse):
        raise ValueError("Calculated values are too big")

    return {
        "coefficient": coefficient,
        "slope": slope,
        "pearson_r": pearson_r,
        "residuals": residuals,
        "mse": mse
    }