import pandas as pd
import matplotlib.pyplot as plt
"""
Create a histogram shoing ditribution of prompt lengths in the dataset
"""

df = pd.read_csv("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/Data/dataset.csv")

# Group prompt lengths by complexity
data = [
    df[df["complexity"] == 0]["prompt_length"],
    df[df["complexity"] == 1]["prompt_length"],
    df[df["complexity"] == 2]["prompt_length"]
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    data,
    widths=0.6,
    patch_artist=True,
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=2)
)

plt.xlabel("Task Complexity", fontsize=12)
plt.ylabel("Prompt Length", fontsize=12)
plt.title("Prompt Length Distribution by Task Complexity", fontsize=14, pad=15)

plt.xticks([1, 2, 3], [0, 1, 2], fontsize=11)
plt.yticks(fontsize=11)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()