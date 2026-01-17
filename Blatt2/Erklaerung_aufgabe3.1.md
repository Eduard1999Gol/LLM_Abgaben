# Aufgabe 1: Die Architektur im Detail erklärt

Wir bauen ein Sprachmodell mit einem simplen Ziel: **Gegeben 3 Wörter (der Kontext), sage das 4. Wort vorher.**

Hier ist der Weg der Daten durch das Netzwerk, Schritt für Schritt.

---

## Schritt 1: Der Input (Das Rohmaterial)

Computer können keine Wörter wie "der", "Hund", "bellt" verstehen. Sie verstehen nur Zahlen.
Deshalb wird in der Vorbereitung ([dataset_vorbereitung.py](dataset_vorbereitung.py)) jedem Wort eine Nummer (Index) gegeben.

**Was kommt rein?**  
Ein Vektor mit 3 Zahlen (den Indizes der Wörter).

**Beispiel:**  
Statt `["der", "hund", "bellt"]` kommt `[42, 105, 12]` rein.

---

## Schritt 2: Der Embedding Layer (Die Übersetzung)

Das ist der wichtigste Teil des Bengio-Papers ("Table look-up in C").

### Das Problem

Die Zahl 42 und 43 sind mathematisch nah beieinander, aber die Wörter dahinter könnten "Apfel" und "Auto" sein (total verschieden). Wir brauchen eine bessere Darstellung als nur eine einfache Zahl.

### Die Lösung (Embedding)

Wir weisen jedem Wort eine Liste von 30 Eigenschaften zu (z.B. Ist es ein Nomen? Ist es belebt? Ist es Plural? – aber das Netz lernt diese Eigenschaften selbst, wir geben sie nicht vor).

### Die Operation

Wir schauen in einer Tabelle nach:

- **Wort 1** $\rightarrow$ Vektor mit 30 Zahlen
- **Wort 2** $\rightarrow$ Vektor mit 30 Zahlen
- **Wort 3** $\rightarrow$ Vektor mit 30 Zahlen

### Exkurs: Wie sieht diese Lookup-Table visuell aus?

Stell dir eine Matrix vor mit Zeilen für jedes Wort und 30 Spalten:

| Index | Wort (Gedanke) | Dim 1  | Dim 2  | ...  | Dim 30 |
|-------|----------------|--------|--------|------|--------|
| 0     | `<UNK>`        | 0.12   | -0.55  | ...  | -0.23  |
| ...   | ...            | ...    | ...    | ...  | ...    |
| 42    | "Hund"         | 0.88   | -0.02  | ...  | 0.44   |
| 43    | "Katze"        | 0.86   | -0.05  | ...  | 0.41   |
| ...   | ...            | ...    | ...    | ...  | ...    |
| 999   | "laufen"       | -0.50  | 0.80   | ...  | 0.05   |

### Was passiert beim "Look-up"?

Wenn der Input "Hund" (Index 42) ist, geht das Modell zur Zeile 42 und kopiert diese 30 Zahlen. Das ist jetzt die Repräsentation für das Wort.

**Der Clou:** Anfangs sind diese Zahlen zufällig. Nach dem Training werden die Zeilen für ähnliche Wörter (wie Hund und Katze) fast identische Zahlen haben, während "laufen" ganz anders aussieht.

### Dimensionalität und Flattening

- **Dimensionalität:** Aus 3 einzelnen Zahlen werden $3 \times 30$ Zahlen
- **Flattening:** Damit das nächste Layer damit arbeiten kann, kleben wir diese drei Vektoren hintereinander zu einer langen Schlange
- **Ergebnis:** Ein Vektor mit 90 Zahlen ($3 \text{ Wörter} \times 30 \text{ Dimensionen}$)

---

## Schritt 3: Der Hidden Layer (Das Gehirn)

Jetzt haben wir 90 Zahlen, die unseren Kontext beschreiben. Jetzt muss das Netz "nachdenken", um Muster zu finden.

### Was passiert hier?

Wir projizieren die 90 Eingangs-Zahlen auf 100 neue Zahlen (Hidden Units). Das passiert durch eine Matrix-Multiplikation.

**Visualisierung:** Jedes der 100 Neuronen im Hidden Layer schaut sich alle 90 Input-Zahlen an, gewichtet sie unterschiedlich und fällt ein Urteil.

### Die Aktivierungsfunktion (Tanh)

- Ohne Aktivierungsfunktion wäre das Netz nur ein einfacher Taschenrechner (linear)
- **Tanh** (Tangens Hyperbolicus) quetscht alle Zahlen in den Bereich zwischen $-1$ und $1$
- Das erlaubt dem Netz, komplexe, nicht-lineare Zusammenhänge zu lernen
  - Beispiel: "Wenn Wort 1 'nicht' ist, kehrt sich die Bedeutung von Wort 3 um"

**Dimensionalität:** Input $90$ $\rightarrow$ Output $100$

---

## Schritt 4: Der Output Layer (Die Vorhersage)

Jetzt hat das Netz "nachgedacht" und hat 100 abstrakte Zahlen im Kopf. Diese müssen wir zurück in Wörter übersetzen.

### Was passiert hier?

Wir wandeln die 100 Hidden-Zahlen in so viele Zahlen um, wie wir Wörter im Vokabular haben (z.B. 5000).

### Die Aktivierungsfunktion (Softmax)

1. Der Output Layer spuckt erst einmal irgendwelche Zahlen aus (**Logits**), z.B. `5.2`, `-1.0`, `12.5`
2. **Softmax** verwandelt diese Zahlen in Wahrscheinlichkeiten, die zusammen 100% (oder 1.0) ergeben
3. Das Wort mit der höchsten Wahrscheinlichkeit ist die Vorhersage des Modells

**Dimensionalität:** Input $100$ $\rightarrow$ Output $|V|$ (Vokabulargröße)

---

## Zusammenfassung: Die "Trainierbaren Parameter"

Wenn wir sagen "das Netz lernt", meinen wir eigentlich: **"Das Netz passt seine Parameter an"**.

Ein Parameter ist einfach eine Zahl, die das Netz verändern darf, um den Fehler zu minimieren.

### Wie viele "Knöpfe" hat das Netz zum Drehen?

(Bei einem Vokabular von z.B. 5000 Wörtern)

#### 1. Im Embedding Layer

Für jedes der 5000 Wörter müssen wir 30 Zahlen lernen.

- **Rechnung:** $5\,000 \times 30 = 150\,000$ Parameter

#### 2. Im Hidden Layer

Jedes der 100 Neuronen ist mit allen 90 Inputs verbunden (Gewichte). Dazu hat jedes Neuron einen Basiswert (Bias).

- **Gewichte:** $90 \times 100 = 9\,000$ Parameter
- **Bias:** $100$ Parameter

#### 3. Im Output Layer

Jedes Wort im Vokabular (5000) ist mit den 100 Hidden-Neuronen verbunden.

- **Gewichte:** $100 \times 5\,000 = 500\,000$ Parameter
- **Bias:** $5\,000$ Parameter

#### Gesamtzahl der Parameter

Das ist die Summe von allem oben. Das ist die Maßzahl für die Komplexität ("Gehirnkapazität") des Modells.

$$P_{\text{total}} = 150\,000 + 9\,000 + 100 + 500\,000 + 5\,000 = 664\,100 \text{ Parameter}$$