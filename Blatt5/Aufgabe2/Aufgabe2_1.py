import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Setze Seed für Reproduzierbarkeit
set_seed(42)

# Modell und Tokenizer laden
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# distilgpt2 hat kein pad_token, wir setzen es auf eos_token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 10 Prompts vorbereiten, auf die "Yoda" antworten soll
prompts = [
    "User: Is the dark side stronger?\nYoda:",
    "User: How do I become a Jedi?\nYoda:",
    "User: I feel fear inside me.\nYoda:",
    "User: Who is the chosen one?\nYoda:",
    "User: What is the Force?\nYoda:",
    "User: Can we trust the clones?\nYoda:",
    "User: Where is Anakin?\nYoda:",
    "User: The Sith are extinct.\nYoda:",
    "User: Should I fight my anger?\nYoda:",
    "User: Are you old?\nYoda:"
]

print("--- 2.1 Ausgaben VOR dem Fine-Tuning ---")
generated_texts_baseline = []

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generierung
    outputs = model.generate(
        **inputs, 
        max_new_tokens=30, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_texts_baseline.append(result)
    print(f"\nPrompt: {prompt.splitlines()[0]}\nOutput: {result.split('Yoda:')[-1].strip()}")
