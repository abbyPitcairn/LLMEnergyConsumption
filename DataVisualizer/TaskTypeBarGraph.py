import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
"""
Create a Bar Graph showing the distribution of the six task types in the dataset.
"""

df = pd.read_csv("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/Data/dataset.csv")

task_counts = df["task_type"].value_counts().sort_index()

plt.figure()
plt.bar(task_counts.index, task_counts.values)
plt.xlabel("Task Type")
plt.ylabel("Number of Prompts")
#plt.title("Distribution of Prompt Task Types")
plt.xticks([1,2,3,4,5,6])

# Custom legend elements
legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', label='1 - Casual'),
    Line2D([0], [0], marker='o', linestyle='None', label='2 - Explanation'),
    Line2D([0], [0], marker='o', linestyle='None', label='3 - Writing'),
    Line2D([0], [0], marker='o', linestyle='None', label='4 - Coding'),
    Line2D([0], [0], marker='o', linestyle='None', label='5 - Productivity'),
    Line2D([0], [0], marker='o', linestyle='None', label='6 - Creative')
]

plt.legend(handles=legend_elements, title="Task Type")
plt.savefig("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/DataVisualizer/graphs/bargraph.pdf")
plt.show()