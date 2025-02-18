import os
import sqlite3
import torch
import ast
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import DatasetDict, Dataset

MODEL_NAME = "models/stablelm_base"
DB_PATH = "tokenized_lyrics.db"

def list_artists(cursor):
  return [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]

def load_tokenized_lyrics(cursor, artist_name):
  """Fetches tokenized lyrics for the given artist from the database and parses tensors correctly."""
  cursor.execute(f"SELECT input_ids FROM {artist_name}")
  rows = cursor.fetchall()

  if not rows:
    raise ValueError(f"No tokenized lyrics found for artist: {artist_name}")

  # Convert input_ids from stored format to a list of integers
  tokenized_texts = []
  for row in rows:
    raw_value = row[0]

    if "tensor" in raw_value:
      # Convert from tensor format
      tensor_values = ast.literal_eval(raw_value.replace("tensor", ""))
      tokenized_texts.append(list(map(int, tensor_values)))
    else:
      # Handle plain integer lists
      tokenized_texts.append(list(map(int, raw_value.split(","))))

  return Dataset.from_dict({"input_ids": tokenized_texts})

def fine_tune_artist(cursor, artist_name):
  """Fine-tunes the model on the selected artist's lyrics."""
  print(f"\n🎤 Starting fine-tuning for artist: {artist_name}...")

  # Load model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map={"": device}  # Explicitly set the device
  )

  # Apply LoRA for efficient fine-tuning
  lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
  )

  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  # Load tokenized dataset
  dataset = DatasetDict({"train": load_tokenized_lyrics(cursor, artist_name)})

  # Training arguments
  save_path = f"models/fine_tuned_{artist_name}"
  training_args = TrainingArguments(
    output_dir=save_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
  )

  # Trainer setup
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    tokenizer=tokenizer
  )

  trainer.train()

  # Save fine-tuned model
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)

  print(f"✅ Model fine-tuned on {artist_name} and saved to {save_path}.")

def main():
  db_conn = sqlite3.connect(DB_PATH)
  cursor = db_conn.cursor()
  cursor.execute("PRAGMA optimize")

  try:
    artists = list_artists(cursor)
    if not artists:
      print("No artist datasets found in the database.")
      return

    print("\nAvailable artists:", ", ".join(artists))
    artist_name = input("Enter the artist name (or type 'all' to fine-tune on all artists): ").strip().lower()

    if artist_name == "all": [fine_tune_artist(cursor, artist) for artist in artists]
    elif artist_name in artists: fine_tune_artist(cursor, artist_name)
    else: print("Invalid artist name. Please check the list and try again.")
    
  finally:
    db_conn.close()  # Ensure the database connection is properly closed

if __name__ == "__main__":
  main()
