# regression.py
# math part of the regression simulator, app.py uses these functions

import math


# average of the values
def mean(values):
    return sum(values) / len(values)


# checks the x and y lists before doing anything
def check_data(x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("X and Y must have the same length")

    if len(x_values) < 2:
        raise ValueError("At least 2 data pairs are required")

    # no inf or nan
    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("X and Y must be finite numbers")


# for log, values have to be above 0
def check_positive(values, message):
    for value in values:
        if value <= 0:
            raise ValueError(message)


# makes sure the answers didn't blow up
def check_results(values):
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Calculated values are too big")


# slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
def compute_slope(x_values, y_values, x_mean, y_mean):
    xy_total = 0
    x_total = 0

    for i in range(len(x_values)):
        x_diff = x_values[i] - x_mean
        y_diff = y_values[i] - y_mean
        xy_total += x_diff * y_diff
        x_total += x_diff ** 2

    # can't divide by 0
    if x_total == 0:
        raise ValueError("All X values are the same")

    return xy_total / x_total


# pearson r formula
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

    # bottom is 0 so r doesn't exist
    if bottom == 0:
        return None

    return xy_total / bottom


# mean squared error
def compute_mse(residuals):
    total = 0

    for residual in residuals:
        total += residual ** 2

    return total / len(residuals)


# linear: y = intercept + slope * x
def predict_linear_y(x, intercept, slope):
    return intercept + slope * x


def run_linear_regression(x_values, y_values):
    check_data(x_values, y_values)

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    try:
        slope = compute_slope(x_values, y_values, x_mean, y_mean)
        intercept = y_mean - slope * x_mean
        pearson_r = compute_pearson_r(x_values, y_values, x_mean, y_mean)

        # residual = actual y - predicted y
        residuals = []
        for i in range(len(x_values)):
            predicted_y = predict_linear_y(x_values[i], intercept, slope)
            residuals.append(y_values[i] - predicted_y)

        mse = compute_mse(residuals)
    except OverflowError:
        raise ValueError("Calculated values are too big")

    check_results([intercept, slope, mse])

    return {
        "intercept": intercept,
        "coefficient": intercept,
        "slope": slope,
        "pearson_r": pearson_r,
        "residuals": residuals,
        "mse": mse
    }


# exponential: y = coefficient * e^(slope * x)
def predict_exponential_y(x, coefficient, slope):
    return coefficient * math.exp(slope * x)


def run_exponential_regression(x_values, y_values):
    check_data(x_values, y_values)
    check_positive(y_values, "Y values must be greater than zero for exponential regression")

    # take ln of y so we can use the linear formulas
    log_y_values = []
    for y in y_values:
        log_y_values.append(math.log(y))

    x_mean = mean(x_values)
    log_y_mean = mean(log_y_values)

    try:
        slope = compute_slope(x_values, log_y_values, x_mean, log_y_mean)
        intercept = log_y_mean - slope * x_mean
        # intercept is ln(a) so use e^ to get a
        coefficient = math.exp(intercept)
        pearson_r = compute_pearson_r(x_values, log_y_values, x_mean, log_y_mean)

        # residuals use the original y
        residuals = []
        for i in range(len(x_values)):
            predicted_y = predict_exponential_y(x_values[i], coefficient, slope)
            residuals.append(y_values[i] - predicted_y)

        mse = compute_mse(residuals)
    except OverflowError:
        raise ValueError("Calculated values are too big")

    check_results([coefficient, slope, mse])

    return {
        "coefficient": coefficient,
        "slope": slope,
        "pearson_r": pearson_r,
        "residuals": residuals,
        "mse": mse
    }


# logarithmic: y = coefficient + slope * ln(x)
def predict_logarithmic_y(x, coefficient, slope):
    return coefficient + slope * math.log(x)


def run_logarithmic_regression(x_values, y_values):
    check_data(x_values, y_values)
    check_positive(x_values, "X values must be greater than zero for logarithmic regression")

    # take ln of x then it's just linear
    log_x_values = []
    for x in x_values:
        log_x_values.append(math.log(x))

    log_x_mean = mean(log_x_values)
    y_mean = mean(y_values)

    try:
        slope = compute_slope(log_x_values, y_values, log_x_mean, y_mean)
        coefficient = y_mean - slope * log_x_mean
        pearson_r = compute_pearson_r(log_x_values, y_values, log_x_mean, y_mean)

        residuals = []
        for i in range(len(x_values)):
            predicted_y = predict_logarithmic_y(x_values[i], coefficient, slope)
            residuals.append(y_values[i] - predicted_y)

        mse = compute_mse(residuals)
    except OverflowError:
        raise ValueError("Calculated values are too big")

    check_results([coefficient, slope, mse])

    return {
        "coefficient": coefficient,
        "slope": slope,
        "pearson_r": pearson_r,
        "residuals": residuals,
        "mse": mse
    }


# 3x3 determinant, used for cramer's rule
def determinant(matrix):
    first_part = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    second_part = matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    third_part = matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])

    return first_part - second_part + third_part


# copy of the matrix with one column swapped for the answers
def replace_column(matrix, column_number, new_values):
    new_matrix = []

    for row_number in range(3):
        new_row = list(matrix[row_number])
        new_row[column_number] = new_values[row_number]
        new_matrix.append(new_row)

    return new_matrix


# quadratic: y = a + bx + cx^2
def predict_quadratic_y(x, a, b, c):
    return a + b * x + c * x * x


# returns a, b, c, residuals, mse and an error message (None if ok)
def fit_quadratic(x_values, y_values):
    point_count = len(x_values)

    # 3 unknowns so need at least 3 points
    if point_count != len(y_values) or point_count < 3:
        return None, None, None, [], None, "You need at least 3 matching x and y values"

    for i in range(point_count):
        if not math.isfinite(x_values[i]) or not math.isfinite(y_values[i]):
            return None, None, None, [], None, "X and Y must be finite numbers"

    # need 3 different x values too
    if len(set(x_values)) < 3:
        return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

    # sums for the normal equations
    sum_x = 0
    sum_x2 = 0
    sum_x3 = 0
    sum_x4 = 0
    sum_y = 0
    sum_xy = 0
    sum_x2y = 0

    for i in range(point_count):
        x = x_values[i]
        y = y_values[i]
        sum_x += x
        sum_x2 += x * x
        sum_x3 += x * x * x
        sum_x4 += x * x * x * x
        sum_y += y
        sum_xy += x * y
        sum_x2y += x * x * y

    main_matrix = [
        [point_count, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]
    ]
    answers = [sum_y, sum_xy, sum_x2y]

    main_determinant = determinant(main_matrix)

    # determinant 0 means no solution
    if main_determinant == 0:
        return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

    # cramer's rule
    a = determinant(replace_column(main_matrix, 0, answers)) / main_determinant
    b = determinant(replace_column(main_matrix, 1, answers)) / main_determinant
    c = determinant(replace_column(main_matrix, 2, answers)) / main_determinant

    residuals = []
    squared_error_total = 0

    for i in range(point_count):
        predicted_y = predict_quadratic_y(x_values[i], a, b, c)
        residual = y_values[i] - predicted_y
        residuals.append(residual)
        squared_error_total += residual * residual

    mse = squared_error_total / point_count

    return a, b, c, residuals, mse, None
