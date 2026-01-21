import torch
import math

# Hilfsfunktion für die Rotation
def rotate_half(x):
    """
    Teilt den Vektor in zwei Hälften.
    Tauscht die Hälften und negiert die erste Hälfte.
    Dies ist ein Trick, um die Multiplikation mit der Rotationsmatrix effizient zu simulieren:
    [-x2, x1] ist das Ergebnis einer 90 Grad Rotation, was Teil der Formel ist.
    """
    x1 = x[..., : x.shape[-1] // 2] # Erste Hälfte der Dimensionen
    x2 = x[..., x.shape[-1] // 2 :] # Zweite Hälfte
    return torch.cat((-x2, x1), dim=-1) 

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Wendet RoPE auf Query (q) und Key (k) an.
    
    Unterschied zu Sinusoidal:
    Bei Sinusoidal (Attention is all you need) passierte dies VOR dem Transformer-Block:
    x = x + sinusoidal_encoding
    
    Bei RoPE passiert dies IM Attention-Head, direkt auf q und k, BEVOR sie multipliziert werden.
    """
    # Die Formel für RoPE: (x * cos) + (rotate_half(x) * sin)
    # Das entspricht der Anwendung der Rotationsmatrix:
    # [x1*cos - x2*sin, x1*sin + x2*cos]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        
        # 1. Berechnung der Frequenzen (Theta)
        # Ähnlich wie beim klassischen Sinusoidal, nutzen wir exponentiell abnehmende Frequenzen.
        # inv_freq entspricht 1 / (10000^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        # Cache für Cos und Sin Werte vorbereiten (für Performance)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        # Erzeugt Positionsindizes [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        # Außenprodukt: Position * Frequenz (m * theta)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Wir duplizieren die Frequenzen, da wir Paare (2D) rotieren
        emb = torch.cat((freqs, freqs), dim=-1)
        # Korrigierte Shape: [1, seq_len, 1, dim] für Broadcasting
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def forward(self, q, k):
        # Dynamisches Update des Caches, falls Sequenz länger ist als erwartet
        seq_len = q.shape[1] # Batch, Seq, Head, Dim (oder ähnlich)
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len
        # Hole die vorbereiteten cos/sin Werte für die aktuelle Sequenzlänge
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return apply_rotary_pos_emb(q, k, cos, sin)

# Beispielhafter Aufruf
dim = 64 # Dimension pro Head
seq_len = 10
rope = RotaryEmbedding(dim=dim, max_seq_len=seq_len)

# Dummy Daten: [Batch=1, Seq=10, Head=4, Dim=64]
# Hinweis: RoPE wird oft auf die Dimensionen [Batch, Seq, Head, Dim] angewendet
q = torch.randn(1, seq_len, 4, dim)
k = torch.randn(1, seq_len, 4, dim)

q_rotated, k_rotated = rope(q, k)

print("Original Norm:", torch.norm(q[0, 0, 0]))
print("Rotated Norm: ", torch.norm(q_rotated[0, 0, 0])) 
# Die Norm sollte nahezu identisch sein (bis auf Rundungsfehler), da es nur eine Rotation ist!