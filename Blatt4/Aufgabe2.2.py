import torch
import torch.nn as nn

def rotate_half(x):
    """
    Hilfsfunktion: Rotiert die Dimensionen des Vektors für die komplexen Operationen.
    Teilt den Vektor in zwei Hälften und tauscht sie mit Vorzeichenwechsel.

    Konkret: Aus [x1, x2] wird [-x2, x1].
    Dies simuliert den Imaginärteil der Multiplikation mit i.
    """
    x1, x2 = x.chunk(2, dim=-1) # Vektor in zwei Hälften teilen (d/2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """
    Wendet die RoPE-Rotation auf die Query- oder Key-Vektoren an.

    Mathematik:
    Die Formel entspricht einer komplexen Rotation: R * x
    Implementiert als: (x * cos) + (rotate_half(x) * sin)

    Args:
        x:  Der Input Tensor (Batch, Seq_Len, Heads, Head_Dim) -> Meist Query oder Key
        cos: Vorberechnete Cosinus-Werte (passend geshaped für Broadcasting)
        sin: Vorberechnete Sinus-Werte (passend geshaped für Broadcasting)
    """
    # UNTERSCHIED ZU SPE (Sinusoidal Positional Encoding):
    # 1. Multiplikativ: Wir addieren die Position nicht (+), wir "mischen" sie hinein (*).
    # 2. Erhaltung der Norm: Da cos^2 + sin^2 = 1, bleibt die Vektorlänge theoretisch stabil,
    #    nur die Orientierung im Raum ändert sich.
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # VORBERECHNUNG (Ähnlich wie bei SPE):
        # Wir nutzen die gleiche geometrische Reihe für Frequenzen wie Vaswani et al.
        # Frequenzen: 10000^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Puffer registrieren (wird nicht trainiert, gehört aber zum State des Modells)
        self.register_buffer("inv_freq", inv_freq)
        
        # Wir berechnen den Cache (cos/sin) einmalig vor, um Rechenzeit zu sparen
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        # Positionen [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Outer Product: Kombiniert Positionen (t) mit Frequenzen (inv_freq)
        # Ergebnis: Matrix [seq_len, dim/2] -> Winkel theta
        freqs = torch.outer(t, self.inv_freq)
        
        # Wir brauchen die Werte für beide Hälften des Vektors (da wir Paare rotieren).
        # Concatenation dupliziert die Winkel, damit sie auf den ganzen Vektor (dim) passen.
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache speichern: [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x, seq_len=None):
        """
        x shape: [Batch, Heads, Seq_Len, Head_Dim]
        (Hinweis: Shape kann je nach Implementation variieren, hier Standard-Layout)
        """
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)

        # Holen der vorberechneten Werte für die aktuelle Sequenzlänge
        # Wir slicen [:seq_len], um nur die relevanten Positionen zu holen.
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # ANWENDUNG:
        # Hier geschieht der entscheidende Schritt.
        # Während SPE typischerweise ganz am Anfang auf die Embeddings addiert wird:
        #   x = TokenEmbedding(x) + PosEmbedding(pos)
        # wird RoPE meistens *innerhalb* der Attention-Schicht angewendet,
        # direkt auf Query (q) und Key (k), bevor der Score berechnet wird.
        return apply_rotary_pos_emb(x, cos, sin)

# --- Beispielaufruf ---

# Parameter
batch_size = 2
heads = 4
seq_len = 10
head_dim = 64 # Dimension pro Head (muss gerade sein für RoPE)

# Dummy Query und Key Tensors (Das, was im Attention Layer passiert)
q = torch.randn(batch_size, heads, seq_len, head_dim)
k = torch.randn(batch_size, heads, seq_len, head_dim)

# Initialisierung des RoPE Moduls
rope = RotaryEmbedding(dim=head_dim)

# Anwenden der Rotation
q_rotated = rope(q, seq_len=seq_len)
k_rotated = rope(k, seq_len=seq_len)

print("Original Q Shape:", q.shape)
print("Rotated Q Shape: ", q_rotated.shape)

# VORTEIL-CHECK (Mathematisch):
# Wenn wir jetzt das Skalarprodukt (Attention Score) berechnen: q_rot @ k_rot.T
# Dann hängt das Ergebnis nur noch von (pos_q - pos_k) ab.
# Bei additiven Encodings (SPE) gäbe es Terme wie (pos_q * pos_k), die nicht rein relativ sind.