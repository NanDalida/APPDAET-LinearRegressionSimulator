# app.py
# GUI for the regression simulator. import a csv, pick x and y and a model,
# then it shows the equation, pearson r, mse and the graph.
# the math is in regression.py

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from regression import (
    run_linear_regression,
    run_exponential_regression,
    run_logarithmic_regression,
    fit_quadratic_regression,
    predict_linear_y,
    predict_exponential_y,
    predict_logarithmic_y,
    predict_quadratic_y,
)

# the csv data, None until a file is imported
df = None


# readonly boxes can't be typed in, so unlock it, change the text, lock it again
def set_readonly_text(entry_widget, text):
    entry_widget.config(state="normal")
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, text)
    entry_widget.config(state="readonly")


# turns a value into a float, None if it's blank, text, inf or nan
def to_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


# gets the rows where both x and y are numbers, counts the ones skipped
def get_valid_pairs(x_col, y_col):
    x_column = df[x_col].tolist()
    y_column = df[y_col].tolist()

    xs = []
    ys = []
    excluded_count = 0

    for i in range(len(x_column)):
        x = to_number(x_column[i])
        y = to_number(y_column[i])

        if x is None or y is None:
            excluded_count += 1
        else:
            xs.append(x)
            ys.append(y)

    return xs, ys, excluded_count


# 101 x values from min to max, used to draw the curve
def make_curve_xs(xs):
    min_x = min(xs)
    max_x = max(xs)

    # all x the same would make the step 0
    if max_x == min_x:
        step = 1
    else:
        step = (max_x - min_x) / 100

    curve_xs = []
    for i in range(101):
        curve_xs.append(min_x + i * step)

    return curve_xs


def format_r(r_value):
    if r_value is None:
        return "N/A"

    return f"{r_value:.4f}"


# each calculate function runs the model, gets the curve points
# and makes the text for the boxes
def calculate_linear(xs, ys):
    result = run_linear_regression(xs, ys)
    intercept = result["intercept"]
    slope = result["slope"]
    mse_value = result["mse"]

    curve_xs = make_curve_xs(xs)
    curve_ys = []
    for x in curve_xs:
        curve_ys.append(predict_linear_y(x, intercept, slope))

    return {
        "equation": f"y = {intercept:.4f} + {slope:.4f}x",
        "r": format_r(result["pearson_r"]),
        "mse": f"{mse_value:.4f}",
        "curve_xs": curve_xs,
        "curve_ys": curve_ys
    }


def calculate_quadratic(xs, ys):
    a, b, c, residuals, mse_value, error = fit_quadratic_regression(xs, ys)

    # fit_quadratic returns the error instead of raising it, so raise it here
    if error is not None:
        raise ValueError(error)

    curve_xs = make_curve_xs(xs)
    curve_ys = []
    for x in curve_xs:
        curve_ys.append(predict_quadratic_y(x, a, b, c))

    return {
        "equation": f"y = {a:.4f} + {b:.4f}x + {c:.4f}x^2",
        "r": "N/A",  # pearson r is only for linear
        "mse": f"{mse_value:.4f}",
        "curve_xs": curve_xs,
        "curve_ys": curve_ys
    }


def calculate_exponential(xs, ys):
    result = run_exponential_regression(xs, ys)
    coefficient = result["coefficient"]
    slope = result["slope"]
    mse_value = result["mse"]

    curve_xs = make_curve_xs(xs)
    curve_ys = []
    for x in curve_xs:
        curve_ys.append(predict_exponential_y(x, coefficient, slope))

    return {
        "equation": f"y = {coefficient:.4f} * e^({slope:.4f}x)",
        "r": format_r(result["pearson_r"]),
        "mse": f"{mse_value:.4f}",
        "curve_xs": curve_xs,
        "curve_ys": curve_ys
    }


def calculate_logarithmic(xs, ys):
    result = run_logarithmic_regression(xs, ys)
    coefficient = result["coefficient"]
    slope = result["slope"]
    mse_value = result["mse"]

    curve_xs = make_curve_xs(xs)
    curve_ys = []
    for x in curve_xs:
        curve_ys.append(predict_logarithmic_y(x, coefficient, slope))

    return {
        "equation": f"y = {coefficient:.4f} + {slope:.4f} * ln(x)",
        "r": format_r(result["pearson_r"]),
        "mse": f"{mse_value:.4f}",
        "curve_xs": curve_xs,
        "curve_ys": curve_ys
    }


# clears the old graph then draws the points and the curve
def draw_graph(xs, ys, output, model, x_col, y_col):
    ax.clear()
    ax.scatter(xs, ys, label="Data Points")
    ax.plot(output["curve_xs"], output["curve_ys"], label=f"{model} Fit")
    ax.set_title(f"{model} Regression")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)
    ax.legend()
    canvas.draw()


# import csv button
def browse_csv():
    global df

    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv")]
    )

    # user cancelled
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        messagebox.showerror(
            "Invalid File",
            "The CSV file could not be read. Make sure it has a header row and one column per variable."
        )
        return

    # put the column names in the dropdowns
    columns = df.columns.tolist()

    x_dropdown["values"] = columns
    y_dropdown["values"] = columns

    # first column is x, second is y by default
    if len(columns) >= 2:
        x_dropdown.current(0)
        y_dropdown.current(1)
    elif len(columns) == 1:
        x_dropdown.current(0)
        y_dropdown.current(0)


# calculate button
def calculate_equation():
    if df is None:
        messagebox.showwarning("Missing Data", "Please import a CSV file first.")
        return

    x_col = x_dropdown.get()
    y_col = y_dropdown.get()
    model = model_dropdown.get()

    if not x_col or not y_col:
        messagebox.showwarning("Missing Columns", "Please select both X and Y columns.")
        return

    xs, ys, excluded_count = get_valid_pairs(x_col, y_col)
    set_readonly_text(excluded_entry, str(excluded_count))

    # need 2 points at least
    if len(xs) < 2:
        messagebox.showerror(
            "Not Enough Data",
            f"Only {len(xs)} valid row(s) found. Rows with a blank or non-numeric X or Y value are excluded."
        )
        return

    # run whichever model is selected
    try:
        if model == "Linear":
            output = calculate_linear(xs, ys)
        elif model == "Quadratic":
            output = calculate_quadratic(xs, ys)
        elif model == "Exponential":
            output = calculate_exponential(xs, ys)
        else:
            output = calculate_logarithmic(xs, ys)
    except ValueError as error:
        messagebox.showerror("Calculation Error", str(error))
        return

    set_readonly_text(equation_entry, output["equation"])
    set_readonly_text(pearson_entry, output["r"])
    set_readonly_text(mse_entry, output["mse"])

    draw_graph(xs, ys, output, model, x_col, y_col)


# Gui Layout

root = tk.Tk()
root.title("Regression Simulator")
root.geometry("950x620")

top_bar = tk.Frame(root, bg="#64748B", height=30)
top_bar.pack(fill="x")

# toolbar with the import button and dropdowns
toolbar = tk.Frame(root, bg="#D1D5DB", height=45)
toolbar.pack(fill="x")

import_button = tk.Button(
    toolbar,
    text="Import CSV",
    command=browse_csv
)
import_button.pack(side="left", padx=5)

x_label = tk.Label(toolbar, text="X:", bg="#D1D5DB")
x_label.pack(side="left", padx=5)

x_dropdown = ttk.Combobox(toolbar, state="readonly", width=14)
x_dropdown.pack(side="left", padx=5)

y_label = tk.Label(toolbar, text="Y:", bg="#D1D5DB")
y_label.pack(side="left", padx=5)

y_dropdown = ttk.Combobox(toolbar, state="readonly", width=14)
y_dropdown.pack(side="left", padx=5)

model_label = tk.Label(toolbar, text="Model:", bg="#D1D5DB")
model_label.pack(side="left", padx=5)

model_dropdown = ttk.Combobox(
    toolbar,
    state="readonly",
    width=14,
    values=["Linear", "Quadratic", "Exponential", "Logarithmic"],
)
model_dropdown.current(0)
model_dropdown.pack(side="left", padx=5)

content_frame = tk.Frame(root, bg="#E5E7EB")
content_frame.pack(fill="both", expand=True)

# left side is the results, right side is the graph
left_panel = tk.Frame(content_frame, bg="#E5E7EB")
left_panel.pack(side="left", fill="both", expand=True)

graph_panel = tk.Frame(
    content_frame,
    bg="white",
    highlightbackground="#D1D5DB",
    highlightthickness=1
)
graph_panel.pack(side="right", fill="both", expand=True, padx=10, pady=5)

# one row per line on the left
row1 = tk.Frame(left_panel, bg="#E5E7EB")
row1.pack(side="top", anchor="w", pady=5)

row2 = tk.Frame(left_panel, bg="#E5E7EB")
row2.pack(side="top", anchor="w", pady=5)

row3 = tk.Frame(left_panel, bg="#E5E7EB")
row3.pack(side="top", anchor="w", pady=5)

row4 = tk.Frame(left_panel, bg="#E5E7EB")
row4.pack(side="top", anchor="w", pady=5)

row5 = tk.Frame(left_panel, bg="#E5E7EB")
row5.pack(side="top", anchor="w", pady=5)

calculate_button = tk.Button(
    row1,
    text="Calculate",
    command=calculate_equation
)
calculate_button.pack(side="left", padx=5)

equation_label = ttk.Label(row2, text="Equation")
equation_label.pack(side="left", padx=5)

equation_entry = ttk.Entry(row2, state="readonly", width=32)
equation_entry.pack(side="left", padx=5)

pearson_label = ttk.Label(row3, text="Pearson R")
pearson_label.pack(side="left", padx=5)

pearson_entry = ttk.Entry(row3, state="readonly", width=32)
pearson_entry.pack(side="left", padx=5)

mse_label = ttk.Label(row4, text="MSE")
mse_label.pack(side="left", padx=5)

mse_entry = ttk.Entry(row4, state="readonly", width=32)
mse_entry.pack(side="left", padx=5)

excluded_label = ttk.Label(row5, text="Excluded Rows")
excluded_label.pack(side="left", padx=5)

excluded_entry = ttk.Entry(row5, state="readonly", width=10)
excluded_entry.pack(side="left", padx=5)

graph_title = tk.Label(
    graph_panel,
    text="Regression Graph",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#1F2937"
)
graph_title.pack(anchor="w", padx=20, pady=15)

# empty graph shown at the start
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")
ax.set_title("Linear Regression")
ax.set_xlabel("X Values")
ax.set_ylabel("Y Values")
ax.grid(True)

# put the matplotlib graph inside the tkinter window
canvas = FigureCanvasTkAgg(fig, master=graph_panel)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
