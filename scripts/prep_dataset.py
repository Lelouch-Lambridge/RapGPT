import os
import sqlite3
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_NAME = "../models/stablelm_base"
DB_FILE = "../datasets.db"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_lyrics_from_db():
  """Extracts all lyrics from the database and structures them."""
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
  tables = [table[0] for table in cursor.fetchall()]
  
  lyrics_data = []
  for table in tables:
    cursor.execute(f"SELECT lyrics FROM {table}")
    lyrics = cursor.fetchall()
    for lyric in lyrics:
      lyrics_data.append({"text": lyric[0]})
  
  conn.close()
  return lyrics_data

# Load lyrics from the database
dataset = Dataset.from_list(get_lyrics_from_db())

def tokenize_function(examples):
  """Tokenizes the lyrics."""
  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Save processed dataset
dataset_dict = DatasetDict({"train": tokenized_dataset})
dataset_dict.save_to_disk("../datasets/tokenized")
print("Dataset tokenized and saved from database.")
