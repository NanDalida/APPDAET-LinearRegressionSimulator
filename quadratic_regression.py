def determinant(matrix):
    first_part = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    second_part = matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    third_part = matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])

    return first_part - second_part + third_part


def replace_column(matrix, column_number, new_values):
    new_matrix = []

    for row_number in range(3):
        new_row = list(matrix[row_number])
        new_row[column_number] = new_values[row_number]
        new_matrix.append(new_row)

    return new_matrix


def fit_quadratic(x_values, y_values):
    point_count = len(x_values)

    if point_count != len(y_values) or point_count < 3:
        return None, None, None, [], None, "You need at least 3 matching x and y values"

    if len(set(x_values)) < 3:
        return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

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

    if main_determinant == 0:
        return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

    a = determinant(replace_column(main_matrix, 0, answers)) / main_determinant
    b = determinant(replace_column(main_matrix, 1, answers)) / main_determinant
    c = determinant(replace_column(main_matrix, 2, answers)) / main_determinant

    residuals = []
    squared_error_total = 0

    for i in range(point_count):
        x = x_values[i]
        y = y_values[i]
        predicted_y = a + b * x + c * x * x
        residual = y - predicted_y
        residuals.append(residual)
        squared_error_total += residual * residual

    mse = squared_error_total / point_count

    return a, b, c, residuals, mse, None
