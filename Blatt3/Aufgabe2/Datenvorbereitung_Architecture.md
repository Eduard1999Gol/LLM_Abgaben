# Datenvorbereitung für Transformer-Modelle: Architecture & Workflow

## Übersicht der Pipeline

Die Datenvorbereitung für Transformer-Modelle erfolgt in drei kritischen Schritten, die aufeinander aufbauen:

---

## Detaillierte Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEP 1: TOKENIZER AUSWAHL                       │
│                              (Aufgabe 2.1)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   Edge Cases testen:          │
                    │   • Umlaute: "Mühlenstraße"   │
                    │   • Emojis: "😊🚀"            │
                    │   • Code: "test_variable"     │
                    │   • Währung: "19.99$"         │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  3 Tokenizer vergleichen:     │
                    │  ✗ BERT (uncased, [UNK])     │
                    │  ✓ GPT-2 (BPE, case-aware)   │
                    │  ✗ T5 (SentencePiece)        │
                    └───────────────┬───────────────┘
                                    │
                            Ergebnis: GPT-2
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: DATASET STATISTIKEN                         │
│                              (Aufgabe 2.2)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  10.000 Samples analysieren:  │
                    │                               │
                    │  1. Zeichenlänge: ~450 chars  │
                    │  2. Token-Länge:  ~100 tokens │
                    │  3. Max Länge:    ~800 tokens │
                    │  4. Vocab Usage:  35k/50k     │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Entscheidungen treffen:     │
                    │                               │
                    │   • block_size = 256          │
                    │   • max_length = 512          │
                    │   • truncation = True         │
                    │   • padding = "max_length"    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 3: BATCH GENERIERUNG                            │
│                              (Aufgabe 2.3)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │                                                         │
        ▼                                                         ▼
┌──────────────────┐                                   ┌──────────────────┐
│  Alle Texte      │                                   │  Zufällige       │
│  tokenisieren    │                                   │  Startpunkte     │
│                  │                                   │  wählen          │
│  "Story 1"       │      ┌──────────────────┐         │                  │
│  "Story 2"  ───► │      │ Full Token Array │   ───►  │  ix = [15, 234]  │
│  "Story 3"       │      │ [1,2,3,...,999]  │         │                  │
└──────────────────┘      └──────────────────┘         └──────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   X und Y Tensoren erstellen  │
                    │                               │
                    │   x = data[i : i+block_size]  │
                    │   y = data[i+1:i+block_size+1]│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                        ╔═══════════════════════╗
                        ║   BATCH OUTPUT:       ║
                        ║                       ║
                        ║   x: [B, T]           ║
                        ║   y: [B, T]           ║
                        ║                       ║
                        ║   B = batch_size      ║
                        ║   T = block_size      ║
                        ╚═══════════════════════╝
```

---

## Schritt-für-Schritt Beschreibung

### 📌 Schritt 1: Tokenizer-Auswahl (Aufgabe 2.1)

#### Ziel
Den optimalen Tokenizer für Textgenerierung finden.

#### Vorgehen
1. **Edge Cases definieren**: 10 Testsätze mit problematischen Zeichen erstellen
2. **Drei Tokenizer testen**:
   - `bert-base-uncased`: BERT Tokenizer
   - `gpt2`: GPT-2 Byte-Level BPE
   - `t5-small`: T5 SentencePiece
3. **Vergleichen**: Wie werden Sonderzeichen, Emojis, Zahlen behandelt?

#### Ergebnis
**GPT-2 gewählt** wegen:
- ✅ **Byte-Level BPE**: Keine Unknown-Tokens
- ✅ **Case-Sensitive**: Erhält Groß-/Kleinschreibung
- ✅ **Decoder-Architecture**: Perfekt für Next-Token Prediction

#### Technische Details
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("Mühlenstraße 😊")
# Ausgabe: ['M', 'ü', 'hlen', 'stra', 'ße', ' ', '😊']
```

---

### 📊 Schritt 2: Dataset-Statistiken (Aufgabe 2.2)

#### Ziel
Die optimale `block_size` und Hyperparameter bestimmen.

#### Vorgehen
1. **10.000 Samples laden** aus TinyStories
2. **Statistiken berechnen**:
   ```python
   lengths_char = []      # Zeichenlänge
   lengths_token = []     # Token-Länge
   vocab_count = set()    # Verwendete Tokens
   ```
3. **Metriken ermitteln**:
   - Durchschnittliche Länge
   - Maximale Länge
   - Vokabular-Coverage

#### Ergebnis
Typische Werte (Beispiel):
- 📏 **Durchschnitt**: 100 Tokens pro Geschichte
- 📈 **Maximum**: 800 Tokens
- 📚 **Vokabular**: 35.000 von 50.257 Tokens genutzt

#### Entscheidungen
```python
block_size = 256      # Kontext-Fenster (basierend auf Durchschnitt)
max_length = 512      # Maximale Sequenzlänge
batch_size = 32       # Parallel verarbeitete Sequenzen
```

#### Warum wichtig?
**Speicher-Komplexität**: O(n²) für Self-Attention
```
block_size = 128  →  16K  Operationen
block_size = 256  →  65K  Operationen (4x mehr!)
block_size = 512  →  262K Operationen (16x mehr!)
```

---

### 🎯 Schritt 3: Batch-Generierung (Aufgabe 2.3)

#### Ziel
Training-Ready Format: (X, Y) Tensoren für Next-Token Prediction.

#### Vorgehen

**1. Alle Texte konkatenieren**
```python
full_text = tokenizer.eos_token.join(data_text_list)
encoded = tokenizer.encode(full_text)
data = torch.tensor(encoded)  # Shape: [N]
```

Beispiel:
```
["Story 1", "Story 2"] 
    ↓
[15, 496, 23, 50256, 39, 1837, 45, 50256]
             ↑               ↑
          EOS-Token       EOS-Token
```

**2. Zufällige Ausschnitte wählen**
```python
ix = torch.randint(len(data) - block_size, (batch_size,))
# ix = [42, 156, 789]  → 3 zufällige Startpunkte
```

**3. X und Y erstellen**
```python
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+block_size+1] for i in ix])
```

#### Die Magie: Y ist X + 1 Position verschoben

```
Text:   "Once upon a time there was"
Tokens: [  1,    2, 3,   4,     5,  6  ]

block_size = 4:

x = [1, 2, 3, 4]  →  Input Sequence
y = [2, 3, 4, 5]  →  Target (um 1 verschoben)

Das Modell lernt:
  Position 0: Input [1]       → Predict 2  ("Once"  → "upon")
  Position 1: Input [1,2]     → Predict 3  ("upon"  → "a")
  Position 2: Input [1,2,3]   → Predict 4  ("a"     → "time")
  Position 3: Input [1,2,3,4] → Predict 5  ("time"  → "there")
```

#### Output Format
```python
x.shape = (batch_size, block_size)  # z.B. (32, 256)
y.shape = (batch_size, block_size)  # z.B. (32, 256)
```

Aus **einem** Batch mit 32 Sequenzen à 256 Tokens erhalten wir:
- **8.192 Trainingsbeispiele** (32 × 256)
- Alle parallel auf GPU verarbeitet! 🚀

---

## Warum diese Architecture?

### 🎯 Autoregressive Prediction
Transformer lernen durch **Next-Token Prediction**:
```
P(w_t | w_1, w_2, ..., w_{t-1})
```

Die Y-Verschiebung gibt uns automatisch die Labels.

### ⚡ Maximum Efficiency
Aus einer Sequenz der Länge N generieren wir N Trainingsbeispiele:
```
Sequenz [1,2,3,4,5]:
  [1]       → predict 2
  [1,2]     → predict 3
  [1,2,3]   → predict 4
  [1,2,3,4] → predict 5
```

### 🔄 Parallelisierung
GPU verarbeitet alle Batch-Elemente gleichzeitig:
```
Batch Size 32: 32× schneller als sequenziell
Batch Size 64: 64× schneller (wenn genug RAM)
```

### 💾 Memory Management
Statt alle Daten im RAM zu halten:
- Streaming-Loading möglich
- Dynamische Batch-Generierung
- Zufälligkeit pro Epoche unterschiedlich

---

## Visualisierung: Von Text zu Tensoren

```
┌─────────────────────────────────────────────────────┐
│  Roher Text                                         │
│  "Once upon a time, there was a brave knight."      │
└─────────────────────┬───────────────────────────────┘
                      │ Tokenizer
                      ▼
┌─────────────────────────────────────────────────────┐
│  Token IDs                                          │
│  [7454, 2402, 257, 640, 11, 612, 373, 257, 14802]  │
└─────────────────────┬───────────────────────────────┘
                      │ get_batch()
                      ▼
┌───────────────────────────────────────┬───────────────────────┐
│  X (Input)                            │  Y (Target)           │
│  [7454, 2402, 257, 640]               │  [2402, 257, 640, 11] │
│  "Once upon a time"                   │  "upon a time ,"      │
└───────────────────────────────────────┴───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Transformer Model                                  │
│  Input: X → Forward Pass → Logits → Loss(Y_pred, Y) │
└─────────────────────────────────────────────────────┘
```
