import pandas as pd
import matplotlib.pyplot as plt
"""
Scatterplot of complexity versus Prompt Length
As we'd expect, complexity has a positive correlation with length
"""

df = pd.read_csv("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/Data/dataset.csv")

plt.figure()
plt.scatter(df["prompt_length"], df["complexity"])

plt.xlabel("Prompt Length")
plt.ylabel("Task Complexity")
#plt.title("Prompt Length vs Task Complexity")

plt.yticks([0, 1, 2])  # Only show these labels on the y-axis

plt.show()