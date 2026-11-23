from traceback import format_stack


def fit_quadratic(x_values, y_values):
    #Check if there is enough matching x and y values
    point_count = len(x_values)
    if point_count !=len(y_values) or point_count <3:
        return None, None, None, [], None, "YOu need at least 3 matching x and y values"

    #initialize values
    sum_x = 0.0
    sum_xsquared = 0.0
    sum_xcubed = 0.0
    sum_xfourth = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xquaredysquared = 0.0

    #loop through each data point to collect sums for the three equations
    for point_index in range(point_count):
        x = x_values[point_index]
        y = y_values[point_index]
        x_squared =  x* x
        sum_x += x
        sum_xsquared += x_squared
        sum_xcubed += x_squared * x
        sum_xfourth += x_squared * x_squared
        sum_y += y
        sum_xquaredysquared += x_squared * y

    #Three Equations for inputs
    first_equation = [float(point_count), sum_x, sum_xsquared, sum_y]
    second_equation = [sum_x, sum_xsquared, sum_xcubed, sum_xy]
    third_equation = [sum_xsquared, sum_xcubed, sum_xfourth, sum_xquaredysquared]

    #Remove a from 2nd and third equation(why) if the number of the first equation is 0 go to the next equation without 0
    if first_equation[0] == 0:
        if second_equation[0] != 0:
            first_equation, second_equation = second_equation, first_equation
        elif third_equation[0] != 0:
            first_equation, third_equation = third_equation, first_equation
        else:
            return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

    #cancel out A in second equation
    factor_for_second = second_equation[0] / first_equation[0]
    for term_position in range(4):
        second_equation[term_position] -= factor_for_second * first_equation[term_position]

    if second_equation[1] == 0:
        return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

    #cancel out A in third equation
    factor_for_third = third_equation[0] / first_equation[0]
    for term_position in range(4):
        third_equation[term_position] -= factor_for_third * first_equation[term_position]

    if third_equation[2] == 0:
        return None, None, None, [], None, "CANNOT solve quadratic equation for these inputs"

    #Solve from c - b - a since its easier
    c = third_equation[3] / third_equation[2]
    b = (second_equation[3] - second_equation[2] * c) / second_equation[1]
    a = (first_equation[3] - first_equation[1] * b - first_equation[2] * c) /first_equation[0]

    residuals = []
    squarederrorsum = 0.0

    #Checks predicted y, resodial error, and sum of squared errors
    for point_index in range(point_count):
        x = x_values[point_index]
        y = y_values[point_index]
        predicted_y = a + (b * x) +  (c * x*x)
        residual = y - predicted_y
        residuals.append(residual)
        squarederrorsum += residual * residual

    mse = squarederrorsum  / point_count
    return a, b, c, residuals, mse, None
