import pandas as pd
import matplotlib.pyplot as plt
"""
Create a stacked bar graph showing the distribution of complexity levels for each task type.
"""

df = pd.read_csv("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/Data/dataset.csv")

# Create a table of counts
counts = pd.crosstab(df["task_type"], df["complexity"])

# Plot stacked bar chart
counts.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.xlabel("Task Type")
plt.ylabel("Number of Prompts")
#plt.title("Task Type vs Task Complexity")

task_names = [
    "Casual", "Explanation", "Writing",
    "Coding", "Productivity", "Creativity"
]

plt.xticks(range(len(task_names)), task_names, rotation=30)
plt.legend(title="Complexity")

plt.tight_layout()
plt.savefig("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/DataVisualizer/graphs/stackedbar.pdf")
plt.show()
