import torch
import math

# --- 1. Implementierung Sinusoidal (Additiv) ---
def get_sinusoidal_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# --- 2. Implementierung RoPE (Multiplikativ/Rotativ) ---
# (Verwendet die Funktionen aus dem vorherigen Schritt)
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_simple(x, pos_idx, dim):
    # Einfache On-the-fly Berechnung für Demo-Zwecke
    # x shape: [1, dim]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.tensor([pos_idx], dtype=torch.float), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return (x * cos) + (rotate_half(x) * sin)

# --- 3. Das Experiment ---

def run_comparison():
    torch.set_printoptions(precision=4, sci_mode=False)
    dim = 64        # Dimension des Head-Vektors
    
    # Wir simulieren zwei Vektoren: Query (Suche) und Key (Inhalt)
    # Zum Beispiel: q="Der", k="Hund"
    q_base = torch.randn(1, dim) # Zufälliger Vektor
    k_base = torch.randn(1, dim) # Zufälliger Vektor
    
    # Vorberechnen der Sinusoidal Encodings (Tabelle)
    spe_table = get_sinusoidal_encoding(200, dim)

    print(f"--- EXPERIMENT START (Dim={dim}) ---\n")

    # === TEST 1: Norm (Vektorlänge) ===
    # Was passiert mit der 'Energie' des Vektors nach der Kodierung?
    
    # SPE (Additiv)
    q_spe = q_base + spe_table[5]
    # RoPE (Rotativ)
    q_rope = apply_rope_simple(q_base, 5, dim)
    
    print(f"1. Vektor-Norm (Länge) bei Position 5:")
    print(f"   Original Norm: {torch.norm(q_base).item():.4f}")
    print(f"   SPE (Additiv): {torch.norm(q_spe).item():.4f} (Verändert!)")
    print(f"   RoPE (Rotativ):{torch.norm(q_rope).item():.4f} (Identisch/Erhalten)")
    print("-" * 40)

    # === TEST 2: Der "Shift-Test" (Relative Position) ===
    # Wir messen die Attention (Skalarprodukt) zwischen q und k.
    # Szenario A: q bei Pos 0, k bei Pos 5 (Abstand 5)
    # Szenario B: q bei Pos 100, k bei Pos 105 (Abstand 5)
    # Erwartung: Ein gutes Modell sollte erkennen, dass der Abstand gleich ist.
    
    print("\n2. Shift-Test (Attention Score q @ k.T):")
    print("   Wir verschieben das Paar (q, k) um 100 Positionen.")
    
    # --- SPE Berechnung ---
    # Pos 0 vs 5
    q_spe_0 = q_base + spe_table[0]
    k_spe_5 = k_base + spe_table[5]
    score_spe_near = torch.sum(q_spe_0 * k_spe_5)
    
    # Pos 100 vs 105
    q_spe_100 = q_base + spe_table[100]
    k_spe_105 = k_base + spe_table[105]
    score_spe_far = torch.sum(q_spe_100 * k_spe_105)
    
    # --- RoPE Berechnung ---
    # Pos 0 vs 5
    q_rope_0 = apply_rope_simple(q_base, 0, dim)
    k_rope_5 = apply_rope_simple(k_base, 5, dim)
    score_rope_near = torch.sum(q_rope_0 * k_rope_5)
    
    # Pos 100 vs 105
    q_rope_100 = apply_rope_simple(q_base, 100, dim)
    k_rope_105 = apply_rope_simple(k_base, 105, dim)
    score_rope_far = torch.sum(q_rope_100 * k_rope_105)

    print(f"\n   [Sinusoidal Additiv]")
    print(f"   Score bei 0->5:    {score_spe_near.item():.4f}")
    print(f"   Score bei 100->105:{score_spe_far.item():.4f}")
    diff_spe = abs(score_spe_near - score_spe_far)
    print(f"   >> Abweichung:     {diff_spe.item():.4f} (Abhängig von absoluter Pos!)")

    print(f"\n   [RoPE Rotativ]")
    print(f"   Score bei 0->5:    {score_rope_near.item():.4f}")
    print(f"   Score bei 100->105:{score_rope_far.item():.4f}")
    diff_rope = abs(score_rope_near - score_rope_far)
    print(f"   >> Abweichung:     {diff_rope.item():.4f} (Perfekt stabil!)")

run_comparison()