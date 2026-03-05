# LLM Energy Consumption - Experimental Research

The aim of this research is to discover the average energy cost of prompting an LLM for text-to-text generation. Currently, this work only generates the dataset to be used. The projected final release date for this project is May 5, 2026.


### Description

First, we develop a dataset representative of the average LLM prompt and then test the energy cost by running this set of prompts through a handful of different LLMs. Finally, these experimental results will be analyzed and a final estimate of energy cost will be given in Watts per hour. 

The dataset is comprised of 500 AI-generated prompts and 500 prompts pulled from verified online datasets. For online datasets, five were selected from the most downloaded page on `HuggingFace.com` with datasets filtered for text-to-text generation and question-answering tasks; then, 100 rows were taken from each dataset using a random seed. These datasets are:

* `HuggingFaceH4/MATH-50`
* `cais/mmlu`
* `openai/openai_humaneval`
* `crownelius/Opus-4.6-Reasoning-3300x`
* `google/simpleqa-verified`

The AI-generated prompts come from ChatGPT's online API and must be generated, saved and uploaded separately as a .csv file. `DatasetGenerator.py` will create the complete dataset from this saved .csv and using the HuggingFace API. 

### Execution

To run the program:

* Download project files. 
* Run the requirements installation and Python commands below. 
* Now your dataset is generated at the output csv file.
  
```
pip install -r requirements.txt
python DatasetGenerator.py
```

### Dataset

In the dataset file, there are six columns: 
* **Prompt**: the prompt text that will be given to the LLM while we measure energy usage.
* **Prompt Length**: the number of words/tokens in the prompt.
* **Task Type**: based on a key with 7 different task types labeled 0-6; includes task categories such as small talk, coding, creativity, etc.
* **Complexity**: a subjective annotation by the author to sort prompts by complexity of the task.
* **Output Length**: model dependent, appended during experiment based on which LLM is being tested at that time.
* **Origin**: the origin of the prompt; either ChatGPT or the name of the HuggingFace dataset.

#### Example of Data Output:

| Prompt | Prompt Length | Task Type | Complexity | Output Length | Origin |
|------|------|------|------|------|------|
| "At 50 miles per hour, how far would a car travel in $2\frac{3}{4}$ hours? Express your answer as a mixed number." | 21 | 2 | 4 | 2 | x | MATH-500 |
| Who was known for playing the trombone for The Jazzmen at the time of Kenny Ball's death? | 17 | 1 | 1 |0 | x | SimpleQA |
| Where should I go on vacation? | 6 | 0 | 1 |0 | x | ChatGPT |

### Authors

* **Lead Author:** Abigail Pitcairn [abigail.pitcairn@maine.edu]

### Release History

* **March 4, 2026:** Initial Release of Dataset Generator
* **May 5, 2026:** Projected Final Release Date
