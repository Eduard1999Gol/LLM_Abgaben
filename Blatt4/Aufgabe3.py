import torch
import torch.nn as nn
from transformers import AutoTokenizer, logging
from datasets import load_dataset
import time

# --- KONFIGURATION ---
# Hier kannst du die Parameter für Aufgabe 2 ändern
CONFIG = {
    'block_size': 32,      # Kontextlänge (wie viele Tokens zurückgeschaut wird)
    'batch_size': 64,      # Wie viele Sequenzen parallel
    'n_embd': 128,          # Dimension der Embeddings (d_model)
    'n_head': 4,           # Anzahl der Attention Heads (muss n_embd teilen: 64/8=8)
    'n_layer': 6,          # Anzahl der Transformer-Blöcke
    'learning_rate': 3e-4,
    'max_iters': 700,      # Wie viele Trainingsschritte insgesamt
    'vocab_size': 50257,   # GPT-2 Vokabulargröße
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Warnung unterdrücken
logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # Fix für Padding

# --- 1. DATEN VORBEREITUNG (aus deinem Skript übernommen & angepasst) ---
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
# Wir nehmen mehr Daten für echtes Training (z.B. 2000 Geschichten)
data_iter = dataset.take(2000) 
text_data = tokenizer.eos_token.join([x['text'] for x in data_iter])
# Einmalige Tokenisierung (dauert kurz)
print("Tokenisiere Daten...")
encoded_data = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)
print(f"Daten geladen. Token-Anzahl: {len(encoded_data)}")

def get_batch():
    ix = torch.randint(len(encoded_data) - CONFIG['block_size'], (CONFIG['batch_size'],))
    x = torch.stack([encoded_data[i:i+CONFIG['block_size']] for i in ix])
    y = torch.stack([encoded_data[i+1:i+CONFIG['block_size']+1] for i in ix])
    return x.to(CONFIG['device']), y.to(CONFIG['device'])

# --- 2. DAS MODELL ---
class MiniTransformerGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Token Embeddings: Wandelt Token-IDs in Vektoren um
        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        # Position Embeddings: Lernt, an welcher Stelle im Satz ein Wort steht
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        
        # Der eigentliche Transformer Block (PyTorch Standard)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['n_embd'], 
            nhead=config['n_head'], 
            dim_feedforward=4*config['n_embd'], # Standard ist 4x d_model
            batch_first=True,
            norm_first=True # Pre-LayerNorm ist stabiler für GPT
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layer'])
        
        # Der Kopf, der zurück auf das Vokabular projiziert
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'])
        self.block_size = config['block_size']

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Embeddings erstellen
        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb
        
        # 2. Causal Mask erstellen (WICHTIG für GPT!)
        # Eine Maske mit -inf oben rechts, damit Tokens nicht in die Zukunft schauen
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        # 3. Durch den Transformer schicken
        x = self.transformer_blocks(x, mask=mask, is_causal=True)
        
        # 4. Logits berechnen
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten für CrossEntropyLoss: (B*T, vocab_size) vs (B*T)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    # Hilfsfunktion zum Generieren von Text
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Kontext auf block_size beschneiden
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Nimm nur den letzten Zeitschritt
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample aus der Verteilung (hier nehmen wir einfach das Wahrscheinlichste -> Greedy)
            # Für mehr Varianz: torch.multinomial(probs, num_samples=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 3. TRAINING ---
print(f"Starte Training auf {CONFIG['device']}...")
model = MiniTransformerGPT(CONFIG).to(CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

# Parameter zählen
num_params = sum(p.numel() for p in model.parameters())
print(f"Anzahl Parameter: {num_params}")

start_time = time.time()
losses = []

for i in range(CONFIG['max_iters']):
    xb, yb = get_batch()
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Step {i}: Loss {loss.item():.4f}")
        losses.append(loss.item())

end_time = time.time()
print(f"Training beendet in {end_time - start_time:.2f} Sekunden.")

# --- 4. GENERIERUNG ---
print("\n--- Generierte Geschichte ---")
start_context = torch.tensor(tokenizer.encode("Once upon a time"), dtype=torch.long, device=CONFIG['device']).unsqueeze(0)
generated_ids = model.generate(start_context, max_new_tokens=50)
print(tokenizer.decode(generated_ids[0].tolist()))