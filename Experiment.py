import pandas as pd
import time
import subprocess
import threading
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Config
DATASET_PATH = "Data/dataset.csv"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # change this per run
OUTPUT_PATH = "results.csv"
MAX_NEW_TOKENS = 100

# Power monitoring function
class PowerMonitor:
    """

    """
    def __init__(self):
        self.running = False
        self.samples = []

    def _monitor(self):
        """
        Uses macOS powermetrics tool to sample power usage.
        Requires sudo privileges.
        """
        cmd = ["sudo", "powermetrics", "--samplers", "smc", "-i", "1000"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while self.running:
            line = process.stdout.readline()
            if not line:
                continue

            # Extract power (look for CPU Power)
            match = re.search(r"CPU Power: (\d+\.?\d*) W", line)
            if match:
                power = float(match.group(1))
                self.samples.append(power)

        process.terminate()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_average_power(self):
        if len(self.samples) == 0:
            return 0
        return sum(self.samples) / len(self.samples)

# Load model
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load dataset
df = pd.read_csv(DATASET_PATH)

results = []

# Run experiment
for _, row in df.iterrows():
    prompt_id = row["id"]
    prompt = row["prompt"]

    print(f"Running prompt {prompt_id}...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Start monitoring
    monitor = PowerMonitor()
    monitor.start()

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    end_time = time.time()
    monitor.stop()

    # Metrics
    response_time = end_time - start_time

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    num_tokens = len(outputs[0]) - inputs["input_ids"].shape[1]

    avg_power = monitor.get_average_power()

    # Energy = Power (W) / Time (hours)
    energy_wh = avg_power / (response_time / 3600)

    results.append({
        "id": prompt_id,
        "output": output_text,
        "num_tokens": num_tokens,
        "avg_watts": avg_power,
        "response_time_sec": response_time,
        "energy_wh": energy_wh
    })

# Save result
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"Results saved to {OUTPUT_PATH}")