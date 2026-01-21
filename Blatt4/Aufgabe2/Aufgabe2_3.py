import torch
import math
from Aufgabe2_2 import RotaryEmbedding 

# Klassisches Sinusoidal Encoding (vereinfacht)
def get_sinusoidal_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Setup
dim = 8 # Kleine Dimension für Übersichtlichkeit
seq_len = 5
base_embedding = torch.ones(seq_len, dim) # Vektor voller Einsen

# 1. Sinusoidal
sinusoidal_pos = get_sinusoidal_encoding(seq_len, dim)
output_sinusoidal = base_embedding + sinusoidal_pos # Additiv

# 2. RoPE
rope_model = RotaryEmbedding(dim)
# Dummy Q (wir nehmen hier an, base_embedding ist unser Q)
# Reshape nötig für RoPE Implementierung [Batch, Seq, Head, Dim]
q_input = base_embedding.unsqueeze(0).unsqueeze(2) 
output_rope, _ = rope_model(q_input, q_input)

print("--- Sinusoidal (Additiv) ---")
print("Norm Token 0:", torch.norm(output_sinusoidal[0]).item())
print("Norm Token 4:", torch.norm(output_sinusoidal[4]).item())

print("\n--- RoPE (Multiplikativ/Rotativ) ---")
print("Norm Token 0:", torch.norm(output_rope[0,0,0]).item())
print("Norm Token 4:", torch.norm(output_rope[0,4,0]).item())