import pandas as pd
import matplotlib.pyplot as plt
"""
Create a histogram showing the distribution of prompt lengths in the dataset. 
"""

df = pd.read_csv("~/Data/dataset.csv")
plt.figure(figsize=(8,5))

plt.hist(df["prompt_length"], bins=20)

plt.xlabel("Prompt Length")
plt.ylabel("Frequency")
#plt.title("Distribution of Prompt Lengths in Dataset")

plt.tight_layout()
plt.savefig("~/DataVisualizer/graphs/histogram.pdf")
plt.show()