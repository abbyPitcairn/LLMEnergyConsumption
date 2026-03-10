import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
"""
Create a word cloud based on the frequency of terms in the prompt texts. 
"""

df = pd.read_csv("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/Data/dataset.csv")

# Combine all prompt text into one string
text = " ".join(df["prompt"].astype(str))

# Generate word cloud
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    max_words=200
).generate(text)

# Plot
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.title("Prompt Word Cloud")
plt.savefig("/Users/abigailpitcairn/Desktop/LLMEnergyConsumption/DataVisualizer/graphs/wordcloud.pdf")
plt.show()