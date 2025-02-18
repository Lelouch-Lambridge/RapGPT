import os
import sqlite3
import sys
import time
from itertools import cycle
from datasets import Dataset
from transformers import AutoTokenizer
from spinner import Spinner

MODEL_NAME = "models/stablelm_base"
DB_FILE = "lyrics.db"
TOKENIZED_DB_FILE = "tokenized_lyrics.db"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def list_tables(cursor):
  """Fetch all table names efficiently."""
  return [name[0] for name in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

def get_lyrics_from_db(cursor, table_name):
  """Retrieve all lyrics in one query to reduce Python loops."""
  return cursor.execute(f"SELECT lyrics FROM {table_name}").fetchall()

def save_tokenized_to_db(cursor, table_name, tokenized_data, connection):
  """Bulk insert tokenized lyrics into the database."""
  cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      input_ids TEXT,
      attention_mask TEXT
    )
  """)

  # Extract input_ids and attention_mask correctly
  input_ids_list = tokenized_data["input_ids"]
  attention_mask_list = tokenized_data["attention_mask"]

  # Convert tokenized data to SQL format
  data_to_insert = [
    (",".join(map(str, input_ids)), ",".join(map(str, attention_mask)))
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list)
  ]

  cursor.executemany(f"INSERT INTO {table_name} (input_ids, attention_mask) VALUES (?, ?)", data_to_insert)
  connection.commit()

  print(f"\r✅ Tokenized data saved to {TOKENIZED_DB_FILE} in table '{table_name}'.    ")


def tokenize_function(text_list):
  """Batch tokenization for efficiency."""
  return tokenizer(text_list, truncation=True, padding="max_length", max_length=256)

def process_table(lyrics_cursor, tokenized_cursor, table_name, tokenized_conn):
  """Processes a single table with optimized SQL queries."""
  print(f"\n🔄 Processing {table_name}...")

  spinner = Spinner.get_instance("Scraping lyrics...")
  spinner.start()

  # Load lyrics in one step instead of using a generator
  lyrics_data = get_lyrics_from_db(lyrics_cursor, table_name)

  if not lyrics_data:
    print(f"\n⚠️ No lyrics found in {table_name}. Skipping.")
    spinner.stop()
    return

  # Extract text for tokenization
  lyrics_texts = [row[0] for row in lyrics_data]

  # Tokenize dataset in a batch operation
  tokenized_dataset = tokenize_function(lyrics_texts)

  # Save tokenized dataset to database
  save_tokenized_to_db(tokenized_cursor, table_name, tokenized_dataset, tokenized_conn)

  spinner.stop()
  sys.stdout.write(f"\r✅ Done processing {table_name}!    \n")
  sys.stdout.flush()

def main():
  """Main function to process lyrics and save tokenized data."""

  # Open database connections once and pass them around
  lyrics_conn = sqlite3.connect(DB_FILE)
  tokenized_conn = sqlite3.connect(TOKENIZED_DB_FILE)
  lyrics_cursor = lyrics_conn.cursor()
  lyrics_cursor.execute("PRAGMA optimize")
  tokenized_cursor = tokenized_conn.cursor()
  tokenized_cursor.execute("PRAGMA optimize")

  try:
    tables = list_tables(lyrics_cursor)
    if not tables:
      print("No tables found in the database. Please check the database file.")
      return

    print("Available artists:", ", ".join(tables))
    table_name = input("Enter the artist name (or type 'all' to process everything): ").strip().lower()

    if table_name == "all":
      for table in tables:
        process_table(lyrics_cursor, tokenized_cursor, table, tokenized_conn)
    elif table_name in tables:
      process_table(lyrics_cursor, tokenized_cursor, table_name, tokenized_conn)
    else:
      print("Invalid table name. Make sure you entered it correctly.")
    
  finally:
    lyrics_conn.close()
    tokenized_conn.close()

if __name__ == "__main__":
  main()
