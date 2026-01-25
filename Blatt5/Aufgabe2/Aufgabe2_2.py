import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv('yoda-corpus.csv')

# Wir bauen Trainingspaare: Wenn YODA spricht, nehmen wir die Zeile davor als "User" Prompt.
training_data = []

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    if row['character'] == 'YODA':
        # Prompt Formatierung
        user_text = prev_row['text']
        yoda_text = row['text']
        
        # Das Format muss konsistent mit dem Prompting in 2.1 sein
        full_text = f"User: {user_text}\nYoda: {yoda_text}{tokenizer.eos_token}"
        training_data.append({"text": full_text})

# In HuggingFace Dataset umwandeln
dataset = Dataset.from_list(training_data)

# Tokenizing Funktion
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(f"\n--- 2.2 Daten vorbereitet: {len(training_data)} Beispiele ---")
print("Beispiel:", training_data[0]['text'])