import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

DATA_DIR = "../datasets"
MODEL_NAME = "../models/stablelm_base"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

artist_files = {
  "Some_Rapper": os.path.join(DATA_DIR, "Some_Rapper_lyrics.txt")
}

# Function to load and structure lyrics
def load_lyrics(artist):
  with open(artist_files[artist], "r", encoding="utf-8") as f:
    lines = f.readlines()
  return {"text": [line.strip() for line in lines if line.strip()]}

dataset_dict = DatasetDict({
  artist: load_dataset("text", data_files={"train": path})["train"]
  for artist, path in artist_files.items()
})

# Tokenization function
def tokenize_function(examples):
  return tokenizer(
    examples["text"], truncation=True, padding="max_length", max_length=128
  )

# Tokenize dataset
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Save processed dataset
tokenized_datasets.save_to_disk("../datasets/tokenized")
print("Dataset tokenized and saved.")
