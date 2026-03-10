import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
"""
Create a scatter plot showing complexity on the x axis and prompt length on the y axis.
"""

df = pd.read_csv("~/Data/dataset.csv")
plt.scatter(df["task_type"], df["prompt_length"])
plt.xlabel("Task Type")
plt.ylabel("Prompt Length (words)")
#plt.title("Task Type vs Prompt Length")

# Custom legend elements
legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', label='1 - Casual'),
    Line2D([0], [0], marker='o', linestyle='None', label='2 - Explanation'),
    Line2D([0], [0], marker='o', linestyle='None', label='3 - Writing'),
    Line2D([0], [0], marker='o', linestyle='None', label='4 - Coding'),
    Line2D([0], [0], marker='o', linestyle='None', label='5 - Productivity'),
    Line2D([0], [0], marker='o', linestyle='None', label='6 - Creative')
]

plt.legend(handles=legend_elements, title="Task Types")
plt.savefig("~/DataVisualizer/graphs/scatterplot.pdf")
plt.show()