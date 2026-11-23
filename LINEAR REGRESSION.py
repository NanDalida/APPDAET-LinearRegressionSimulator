import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Linear Regression Simulator")
root.geometry("900x600")

top_bar = tk.Frame(root, bg="#64748B", height=30)
top_bar.pack(fill="x")

toolbar = tk.Frame(root, bg="#D1D5DB", height=45)
toolbar.pack(fill="x")


def browse_csv():
    global df

    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv")]
    )

    if file_path:
        df = pd.read_csv(file_path)
        print(df.head())

        columns = df.columns.tolist()

        x_dropdown["values"] = columns
        y_dropdown["values"] = columns

        x_dropdown.current(0)
        y_dropdown.current(1)
    
import_button = tk.Button(
    toolbar,
    text="Import CSV",
    command=browse_csv
)
import_button.pack(side="left", padx=5)

x_label = tk.Label(toolbar, text="X:", bg="#D1D5DB")
x_label.pack(side="left", padx=5)

x_dropdown = ttk.Combobox(toolbar, state="readonly")
x_dropdown.pack(side="left", padx=5)

y_label = tk.Label(toolbar, text="Y:", bg="#D1D5DB")
y_label.pack(side="left", padx=5)

y_dropdown = ttk.Combobox(toolbar, state="readonly")
y_dropdown.pack(side="left", padx=5)

content_frame = tk.Frame(root, bg="#E5E7EB")
content_frame.pack(fill="both", expand=True)

left_panel = tk.Frame(content_frame, bg="#E5E7EB")
left_panel.pack(side="left", fill="both", expand=True)

graph_panel = tk.Frame(
    content_frame,
    bg="white",
    highlightbackground="#D1D5DB",
    highlightthickness=1
)
graph_panel.pack(side="right", fill="both", expand=True, padx=10, pady=5)

row1 = tk.Frame(left_panel, bg="#E5E7EB")
row1.pack(side="top", anchor="w", pady=5)

row2 = tk.Frame(left_panel, bg="#E5E7EB")
row2.pack(side="top", anchor="w", pady=5)

row3 = tk.Frame(left_panel, bg="#E5E7EB")
row3.pack(side="top", anchor="w", pady=5)

row4 = tk.Frame(left_panel, bg="#E5E7EB")
row4.pack(side="top", anchor="w", pady=5)

def calculate_equation():
    pass

calculate = tk.Button(
    row1,
    text="Calculate",
    command=calculate_equation
)
calculate.pack(side="left", padx=5)

Equation = ttk.Label(row2, text="Equation")
Equation.pack(side="left", padx=5)

equation = ttk.Entry(row2, state="readonly")
equation.pack(side="left", padx=5)

Pearson = ttk.Label(row3, text="Pearson R")
Pearson.pack(side="left", padx=5)

pearson = ttk.Entry(row3, state="readonly")
pearson.pack(side="left", padx=5)

mse = ttk.Label(row4, text="MSE")
mse.pack(side="left", padx=5)

mse1 = ttk.Entry(row4, state="readonly")
mse1.pack(side="left", padx=5)

graph_title = tk.Label(
    graph_panel,
    text="Regression Graph",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#1F2937"
)
graph_title.pack(anchor="w", padx=20, pady=15)

fig, ax = plt.subplots(figsize=(7, 5))

fig.patch.set_facecolor("white")

ax.set_title("Linear Regression")
ax.set_xlabel("X Values")
ax.set_ylabel("Y Values")
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=graph_panel)
canvas.draw()

canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
