from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "stabilityai/stablelm-zephyr-3b"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  torch_dtype="auto",
  device_map="auto"
)

# Save to local
model.save_pretrained("./models/stablelm_base")
tokenizer.save_pretrained("./models/stablelm_base")

print("StableLM model downloaded and saved.")
