import torch
from transformers import AutoTokenizer


# Wir nehmen den GPT2 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_batch(data_text_list, block_size, batch_size):
    """
    data_text_list: Liste von Strings (den Geschichten)
    block_size: Wie viele Token pro Sequenz (dein "Satz Länge 3")
    batch_size: Wie viele Sequenzen parallel verarbeitet werden
    """
    # 1. Alles tokenisieren und in einen riesigen 1D-Tensor packen
    # (In der Praxis macht man das einmal vorher und speichert es auf Disk)
    full_text = tokenizer.eos_token.join(data_text_list)
    encoded = tokenizer.encode(full_text)
    data = torch.tensor(encoded, dtype=torch.long)
    
    # 2. Zufällige Startpunkte im Text auswählen
    # Wir brauchen Platz für block_size + 1 (weil y um 1 verschoben ist)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 3. Stücke ausschneiden
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

# --- TEST ---
test_stories = [
    "This is a story about a cat.",
    "The dog barked loud.",
    "Once upon a time there was a code."
]

# Sagen wir, wir wollen Sequenzen der Länge 4 (dein Beispiel "Batch-size" von 3, hier 4)
BLOCK_SIZE = 4 
BATCH_SIZE = 2 # Wir wollen 2 Beispiele gleichzeitig sehen

xb, yb = get_batch(test_stories, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

print(f"Shape von x: {xb.shape} (Batch Size, Block Size)")
print("-" * 30)

for b in range(BATCH_SIZE):
    print(f"\nBatch {b+1}:")
    print(f"x (Input):  {xb[b].tolist()} -> {[tokenizer.decode([t]) for t in xb[b]]}")
    print(f"y (Target): {yb[b].tolist()} -> {[tokenizer.decode([t]) for t in yb[b]]}")