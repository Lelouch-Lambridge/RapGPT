from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

def load_model(model_path):
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(model_path)
  return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_rap(generator, prompt):
  lyrics = generator(prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, num_return_sequences=1)
  return lyrics[0]["generated_text"]

def main(selected_model, prompt):
  generator = load_model(selected_model)
  generated_lyrics = generate_rap(generator, prompt)

  output_file = "output/generated_lyrics.txt"
  with open(output_file, "w") as f:
    f.write(generated_lyrics)

  print("Generated Lyrics:\n", generated_lyrics)
  print(f"Lyrics saved to {output_file}")

if __name__ == "__main__":
  models_dir = "models"
  available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
  
  if not available_models:
    print("No models found in the models directory.")
    exit()

  print("Available models:")
  for idx, model in enumerate(available_models, start=1):
    print(f"{idx}. {model}")
  
  while True:
    try:
      choice = int(input("Select a model by number: "))
      if 1 <= choice <= len(available_models):
        selected_model = os.path.join(models_dir, available_models[choice - 1])
        break
      else:
        print("Invalid choice. Try again.")
    except ValueError:
      print("Please enter a valid number.")
  
  prompt = input("Enter a prompt for lyric generation: ")
  main(selected_model, prompt)
