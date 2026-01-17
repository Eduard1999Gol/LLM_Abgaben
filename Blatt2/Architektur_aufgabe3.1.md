# Architektur des Neural Probabilistic Language Model (NPLM)

Analyse der Architektur nach Bengio et al.

---

## 1. Die Architektur-Pipeline

Das Netzwerk verarbeitet die Daten in folgenden Schritten:

### 1.1 Input Layer (Kontext)

- **Input:** 3 Indizes (Wort IDs)
- **Dimension:** $3$ (Vektor mit 3 Ganzzahlen)

### 1.2 Embedding Layer (C) - "Table look-up"

- Jeder der 3 Indizes wird in einen Vektor der Größe 30 umgewandelt
- Da wir 3 Wörter haben, erhalten wir 3 Vektoren à 30 Dimensionen
- Diese werden aneinandergereiht (konkateniert)
- **Dimension nach Flattening:** $3 \times 30 = 90$

### 1.3 Hidden Layer (H)

- Verbindet die 90 Inputs mit 100 Neuronen
- **Aktivierungsfunktion:** tanh
- **Input-Dimension:** $90$
- **Output-Dimension:** $100$

### 1.4 Output Layer (O)

- Projiziert die 100 Hidden-Units auf die Größe des gesamten Vokabulars $|V|$
- **Aktivierungsfunktion:** LogSoftmax (für NLL Loss)
- **Output-Dimension:** $|V|$ (Wahrscheinlichkeit für jedes Wort im Vokabular)

---

## 2. Berechnung der Trainierbaren Parameter

### Notation

- $|V|$ = Größe des Vokabulars (z.B. 5000 Wörter)
- $m = 30$ = Embedding-Größe
- $c = 3$ = Kontextlänge
- $h = 100$ = Anzahl Hidden-Units

### 2.1 Embedding Matrix ($C$)

Dies ist die Lookup-Tabelle. Sie speichert für jedes Wort im Vokabular einen Vektor.

- **Form:** $|V| \times m$
- **Parameter:** $|V| \times 30$

### 2.2 Hidden Layer: Gewichte ($W_1$) & Bias ($b_1$)

Hier verbinden wir den konkatenierten Kontext ($3 \times 30 = 90$) mit den 100 Hidden Units.

- **Gewichtsmatrix:** $90 \times 100$ (Input $\to$ Hidden)
  - Parameter: $9\,000$
- **Bias Vektor:** $100$
  - Parameter: $100$
- **Summe Layer 1:** $9\,100$ Parameter

### 2.3 Output Layer: Gewichte ($W_2$) & Bias ($b_2$)

Verbindung von Hidden Units zum Vokabular.

- **Gewichtsmatrix:** $100 \times |V|$
  - Parameter: $100 \times |V|$
- **Bias Vektor:** $|V|$
  - Parameter: $|V|$
- **Summe Layer 2:** $101 \times |V|$ Parameter

### 2.4 Gesamtsumme ($P_{\text{total}}$)

$$P_{\text{total}} = (|V| \times 30) + 9\,100 + (101 \times |V|)$$

$$P_{\text{total}} = 131 \times |V| + 9\,100$$

#### Beispiel

Wenn das Vokabular 1000 Wörter umfasst:

$$P_{\text{total}} = 131 \times 1\,000 + 9\,100 = 140\,100 \text{ Parameter}$$