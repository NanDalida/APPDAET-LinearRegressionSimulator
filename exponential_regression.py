import math

# get the average
def mean(values):
    return sum(values) / len(values)

# slope
def compute_slope(xs, log_ys, x_mean, y_mean):
    numerator = 0
    denominator = 0

    for i in range(len(xs)):
        x_diff = xs[i] - x_mean
        y_diff = log_ys[i] - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff ** 2

    if denominator == 0:
        raise ValueError("All X values are the same")

    return numerator / denominator

# coefficient
def compute_coefficient(x_mean, y_mean, slope):
    intercept = y_mean - slope * x_mean
    return math.exp(intercept)

# pearson r
def compute_pearson_r(xs, log_ys, x_mean, y_mean):
    numerator = 0
    x_total = 0
    y_total = 0

    for i in range(len(xs)):
        x_diff = xs[i] - x_mean
        y_diff = log_ys[i] - y_mean
        numerator += x_diff * y_diff
        x_total += x_diff ** 2
        y_total += y_diff ** 2

    denominator = math.sqrt(x_total * y_total)
    if denominator == 0:
        return None

    return numerator / denominator

#-------------------------------------------

# predict y using x
def predict_y(x, coefficient, slope):
    return coefficient * math.exp(slope * x)

# difference between actual and predicted y
def compute_residuals(xs, ys, coefficient, slope):
    residuals = []

    for i in range(len(xs)):
        predicted = predict_y(xs[i], coefficient, slope)
        residuals.append(ys[i] - predicted)

    return residuals

# mean squared error
def compute_mse(residuals):
    total = 0

    for residual in residuals:
        total += residual ** 2

    return total / len(residuals)

# exponential regression calculations
def run_exponential_regression(xs, ys):
  
    # check if x and y have enough matching values
    if len(xs) != len(ys):
        raise ValueError("X and Y must have the same length")

    if len(xs) < 2:
        raise ValueError("At least 2 data pairs are required")

    log_ys = []

    # check values and convert y to natural log
    for i in range(len(xs)):
        if not math.isfinite(xs[i]) or not math.isfinite(ys[i]):
            raise ValueError("X and Y must be finite numbers")

        if ys[i] <= 0:
            raise ValueError("Y values must be greater than zero")

        log_ys.append(math.log(ys[i]))

    # get the averages and slope
    x_mean = mean(xs)
    y_mean = mean(log_ys)
    slope = compute_slope(xs, log_ys, x_mean, y_mean)

    try:
        # calculates coefficient and pearson r
        coefficient = compute_coefficient(x_mean, y_mean, slope)
        pearson_r = compute_pearson_r(xs, log_ys, x_mean, y_mean)

        # calculates residuals and mse
        residuals = compute_residuals(xs, ys, coefficient, slope)
        mse = compute_mse(residuals)

    except OverflowError:
        raise ValueError("Calculated values are too big")

    if not math.isfinite(coefficient) or not math.isfinite(slope) or not math.isfinite(mse):
        raise ValueError("Calculated values are too big")

    # results
    return {
        "coefficient": coefficient,
        "slope": slope,
        "pearson_r": pearson_r,
        "residuals": residuals,
        "mse": mse
    }
