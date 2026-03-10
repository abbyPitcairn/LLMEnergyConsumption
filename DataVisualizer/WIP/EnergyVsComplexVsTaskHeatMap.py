import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Coming Soon, this is an idea for post-experimentation data
"""

# Load dataset
df = pd.read_csv("dataset.csv")

# Create pivot table
heatmap_data = df.pivot_table(
    values="energy_joules",
    index="task_type",
    columns="complexity",
    aggfunc="mean"
)

# Convert to numpy array
data = heatmap_data.values

plt.figure()

plt.imshow(data)

plt.colorbar(label="Average Energy Consumption (Joules)")

plt.xticks(
    ticks=np.arange(len(heatmap_data.columns)),
    labels=["Low", "Medium", "High"]
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index
)

plt.xlabel("Task Complexity")
plt.ylabel("Task Type")

plt.title("Average Energy Consumption by Task Type and Complexity")

plt.show()