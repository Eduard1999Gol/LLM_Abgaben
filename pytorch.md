# PyTorch Cheat Sheet für Einsteiger

---

## 1. Tensoren (Die Datenbasis)

Tensoren sind wie NumPy-Arrays, können aber auf der GPU laufen und speichern Gradienten für das Training.

### Erstellung

```python
import torch

# Aus Liste erstellen
x = torch.tensor([1, 2, 3])

# Mit festen Werten
x = torch.zeros(3, 4)       # 3x4 Matrix mit Nullen
x = torch.ones(3, 4)        # 3x4 Matrix mit Einsen
x = torch.randn(3, 4)       # Zufallswerte (Normalverteilung)
x = torch.arange(0, 10)     # [0, 1, ..., 9]
```

### Informationen & Umformung

```python
x.shape                     # Gibt Dimensionen zurück (z.B. torch.Size([3, 4]))
x.device                    # Wo liegt der Tensor? (cpu oder cuda:0)

x.view(12)                  # Flachklopfen zu Vektor der Länge 12
x.view(-1, 4)               # Automatische Dimensionierung (hier: 3 Zeilen, 4 Spalten)
x.unsqueeze(0)              # Fügt Dimension an Index 0 hinzu (aus [3] wird [1, 3])
x.squeeze()                 # Entfernt alle Dimensionen der Größe 1
```

### Konvertierung

```python
x.item()                    # Holt Wert aus Skalar-Tensor (nur bei 1 Element möglich!)
x.numpy()                   # Wandelt Tensor in NumPy Array um (nur auf CPU möglich)
torch.from_numpy(np_arr)    # Wandelt NumPy Array in Tensor um
```

---

## 2. Neural Networks (torch.nn)

Das Modul für Schichten und Loss-Funktionen.

### Wichtige Layer

```python
import torch.nn as nn

# Lineare Schicht (Fully Connected): y = xW + b
# Input-Größe -> Output-Größe
layer = nn.Linear(in_features=10, out_features=5)

# Aktivierungsfunktionen
relu = nn.ReLU()            # Setzt negative Werte auf 0
sigmoid = nn.Sigmoid()      # Quetscht Werte zwischen 0 und 1
softmax = nn.Softmax(dim=1) # Wahrscheinlichkeitsverteilung (Summe = 1)
```

### Loss Funktionen (Fehlerberechnung)

- **Klassifikation:** `nn.CrossEntropyLoss()` (Erwartet Logits, kein Softmax vorher!)
- **Regression:** `nn.MSELoss()` (Mean Squared Error für Zahlenvorhersage)

### Modell-Grundgerüst

Jedes eigene Netz muss von `nn.Module` erben.

```python
class MeinNetz(nn.Module):
    def __init__(self):
        super().__init__()
        # Hier Layer definieren
        self.layer1 = nn.Linear(10, 20)
        self.output = nn.Linear(20, 1)
        
    def forward(self, x):
        # Hier den Datenfluss definieren
        x = torch.relu(self.layer1(x))
        return self.output(x)
```

---

## 3. Training & Autograd (Der Motor)

PyTorch merkt sich alle Rechenoperationen, um später ableiten zu können (Backpropagation).

### Der Trainings-Loop (Die 5 heiligen Schritte)

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 1. Forward Pass: Vorhersage berechnen
output = model(input_data)

# 2. Loss berechnen: Fehler bestimmen
loss = criterion(output, target_data)

# 3. Alte Gradienten löschen (WICHTIG!)
optimizer.zero_grad()

# 4. Backward Pass: Gradienten berechnen (dLoss/dx)
loss.backward()

# 5. Gewichte aktualisieren
optimizer.step()
```

### Modus umschalten

Manche Layer (z.B. Dropout, BatchNorm) verhalten sich beim Training anders als beim Testen.

- `model.train()`: Aktiviert Trainings-Verhalten (Standard)
- `model.eval()`: Aktiviert Test-Verhalten (Wichtig vor Validierung/Test!)

### Gradienten deaktivieren

Spart Speicher und Rechenzeit bei der Vorhersage (Inferenz):

```python
with torch.no_grad():
    prediction = model(test_data)
```

---

## 4. Wichtige Regeln & Fehlerquellen

### ⚠️ Shape Mismatch

> [!danger]
> Die häufigste Fehlerquelle! Überprüfe immer `.shape` von Input und Gewichten.
> Bei `nn.Linear(A, B)` muss der Input in der letzten Dimension die Größe A haben.

### 🔧 Device Mismatch

> [!warning]
> Tensoren und Modell müssen auf dem gleichen Gerät sein (CPU oder GPU).
> 
> **Fehler:** `Expected all tensors to be on the same device.`
> 
> **Lösung:** `data = data.to('cuda')` und `model = model.to('cuda')`

### 📊 CrossEntropyLoss Besonderheit

> [!info]
> **Input:** Erwartet rohe Scores (Logits) vom Netz – also **kein** Softmax am Ende des Modells!
> 
> **Target:** Erwartet Klassen-Indizes als `long` (z.B. `3`), nicht One-Hot-Encoded.

### ✅ Zero Grad & Forward

> [!important]
> - Vergiss nie `optimizer.zero_grad()`, sonst werden Gradienten addiert (Accumulation)
> - Rufe das Modell immer direkt auf: `model(x)` (intern wird `forward` genutzt), nicht `model.forward(x)`

---
