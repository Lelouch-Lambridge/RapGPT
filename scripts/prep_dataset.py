import os
import sqlite3
import sys
import time
import threading
from itertools import cycle
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "models/stablelm_base"
DB_FILE = "lyrics.db"
TOKENIZED_DB_FILE = "tokenized_lyrics.db"

SPINNER_FRAMES = cycle(["⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇"])
SPINNER_RUNNING = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def spinner():
  """Display a rotating spinner while processing data."""
  while SPINNER_RUNNING:
    sys.stdout.write(f"\rProcessing... {next(SPINNER_FRAMES)} ")
    sys.stdout.flush()
    time.sleep(0.1)

def list_tables():
  """Lists available tables in the database."""
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
  tables = [table[0] for table in cursor.fetchall()]
  conn.close()
  return tables

def get_lyrics_from_db(table_name):
  """Extracts lyrics from the selected table in the database."""
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(f"SELECT lyrics FROM {table_name}")
  lyrics = [row[0] for row in cursor.fetchall()]
  conn.close()
  return [{"text": lyric} for lyric in lyrics]

def save_tokenized_to_db(table_name, tokenized_data):
  """Saves tokenized lyrics to a new database."""
  conn = sqlite3.connect(TOKENIZED_DB_FILE)
  cursor = conn.cursor()

  # Create table if not exists
  cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      input_ids TEXT,
      attention_mask TEXT
    )
  """)

  # Insert tokenized data
  for entry in tokenized_data:
    input_ids_str = ",".join(map(str, entry["input_ids"]))
    attention_mask_str = ",".join(map(str, entry["attention_mask"]))
    cursor.execute(f"INSERT INTO {table_name} (input_ids, attention_mask) VALUES (?, ?)", (input_ids_str, attention_mask_str))

  conn.commit()
  conn.close()
  print(f"\r✅ Tokenized data saved to {TOKENIZED_DB_FILE} in table '{table_name}'.    ")

def tokenize_function(examples):
  """Tokenizes the lyrics."""
  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

def process_table(table_name):
  """Processes a single table: loads, tokenizes, and saves data."""
  print(f"\n🔄 Processing {table_name}...")

  # Start spinner
  global SPINNER_RUNNING
  SPINNER_RUNNING = True
  spinner_thread = threading.Thread(target=spinner, daemon=True)
  spinner_thread.start()

  # Load lyrics from the selected database table
  dataset = Dataset.from_list(get_lyrics_from_db(table_name))

  # Tokenize dataset
  tokenized_dataset = dataset.map(tokenize_function, batched=True)
  tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

  # Save tokenized dataset to database
  save_tokenized_to_db(table_name, tokenized_dataset)

  # Stop spinner
  SPINNER_RUNNING = False
  spinner_thread.join()

  sys.stdout.write(f"\r✅ Done processing {table_name}!    \n")
  sys.stdout.flush()

def main():
  """Main function to process lyrics and save tokenized data."""
  tables = list_tables()
  if not tables:
    print("No tables found in the database. Please check the database file.")
    return

  print("Available artists:", ", ".join(tables))
  table_name = input("Enter the artist name (or type 'all' to process everything): ").strip().lower()

  if table_name == "all": [ process_table(table) for table in tables ]
  elif table_name in tables: process_table(table_name)
  else: print("Invalid table name. Make sure you entered it correctly.")

if __name__ == "__main__":
  main()
