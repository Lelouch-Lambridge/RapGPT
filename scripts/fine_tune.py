import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

MODEL_NAME = "../models/stablelm_base"
DATASET_PATH = "../datasets/tokenized"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  torch_dtype="auto",
  device_map="auto"
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
dataset = load_from_disk(DATASET_PATH)

# Training arguments
training_args = TrainingArguments(
  output_dir="../models/Some_Model",
  evaluation_strategy="epoch",
  save_strategy="epoch",
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  num_train_epochs=3,
  weight_decay=0.01,
  logging_dir="../logs",
  logging_steps=100,
  save_total_limit=2,
  push_to_hub=False,
  fp16=torch.cuda.is_available(),
)

# Trainer setup
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=dataset["Some_Dataset"],
  eval_dataset=dataset["Some_Dataset"],
  tokenizer=tokenizer
)

trainer.train()

# Save fine-tuned model
model.save_pretrained("../models/Some_Model")
tokenizer.save_pretrained("../models/Some_Model")

print("Model fine-tuned and saved.")
