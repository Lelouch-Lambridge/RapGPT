from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "../models/Some_Model"

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_rap(prompt):
  lyrics = generator(prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, num_return_sequences=1)
  return lyrics[0]["generated_text"]

# Example usage
prompt = "Something??"
generated_lyrics = generate_rap(prompt)

# Save to file
with open("../output/generated_lyrics.txt", "w") as f:
  f.write(generated_lyrics)

print("Generated Lyrics:\n", generated_lyrics)
