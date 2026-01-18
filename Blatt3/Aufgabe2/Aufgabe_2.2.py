from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import os

# Disable parallelism to avoid threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Datensatz laden (nur den Train-Split, um Zeit zu sparen)
# TinyStories ist groß, wir nehmen hier exemplarisch die ersten 10.000 Einträge für die Statistik
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
data_iter = dataset.take(10000) 

# Tokenizer (den wir oben gewählt haben)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT2 hat kein Pad-Token standardmäßig, wir setzen es auf EOS für Statistiken
tokenizer.pad_token = tokenizer.eos_token 

lengths_char = []
lengths_token = []
vocab_count = set() # Einfaches Set für Unique Words (grobe Schätzung)

print("Berechne Statistiken (das kann kurz dauern)...")
try:
    for item in data_iter:
        text = item['text']
        
        # 1. Statistik: Länge in Zeichen
        lengths_char.append(len(text))
        
        # Tokenizing für genauere Stats
        # Die Token-Länge bestimmt deine spätere block_size (Kontext-Fenster). 
        # Wenn die meisten Geschichten 200 Token lang sind, 
        # reicht ein Kontext von 256. Sind sie 1000 lang, brauchst du mehr Speicher.
        tokens = tokenizer(text, truncation=True, max_length=1024)['input_ids']
        
        # 2. Statistik: Länge in Tokens
        lengths_token.append(len(tokens))
        
        # 3. Statistik: Wortschatz-Coverage (Unique Token IDs in diesem Sample)
        vocab_count.update(tokens)
except Exception as e:
    print(f"Fehler beim Verarbeiten: {e}")
    pass

# Ergebnisse
print(f"--- Statistiken (basierend auf 10.000 Samples) ---")
print(f"1. Durchschnittliche Länge (Zeichen): {np.mean(lengths_char):.2f}")
print(f"2. Durchschnittliche Länge (Tokens):  {np.mean(lengths_token):.2f}")
print(f"3. Max Länge (Tokens):                {np.max(lengths_token)}")
print(f"4. Genutzte Unique Tokens (Sample):   {len(vocab_count)} (von {tokenizer.vocab_size} möglichen)")