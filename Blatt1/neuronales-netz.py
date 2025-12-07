# filepath: /home/eduard/Desktop/LLM/Blatt1/neuronales-netz.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# Textdatei einlesen
with open('extracted_text.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Tokenisierung: Text in Wörter aufteilen
words = text.replace('\n', ' ').replace('.', ' ').split()
words = [w.strip() for w in words if w.strip()]

# Vokabular erstellen (alle einzigartigen Wörter)
vocab = sorted(set(words))
vocab_size = len(vocab)

print(f"Anzahl der Wörter im Text: {len(words)}")
print(f"Anzahl einzigartiger Wörter: {vocab_size}")
print(f"Vokabular: {vocab}")

# Wort zu Index und Index zu Wort Mapping
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

# Trainingsdaten erstellen: Paare von aufeinanderfolgenden Wörtern
X_train = []
y_train = []

for i in range(len(words) - 1):
    input_word = words[i]
    output_word = words[i + 1]
    
    # One-Hot Encoding für Input
    input_vec = torch.zeros(vocab_size)
    input_vec[word_to_idx[input_word]] = 1.0
    X_train.append(input_vec)
    
    # Label für Output (Index des nächsten Wortes)
    y_train.append(word_to_idx[output_word])

X_train = torch.stack(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

print(f"\nTrainingsdaten erstellt: {len(X_train)} Beispiele")

# Neuronales Netz definieren
class SimpleWordNet(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleWordNet, self).__init__()
        # Einfaches Netz: Input Layer -> Output Layer (direkte Verbindung)
        self.linear = nn.Linear(vocab_size, vocab_size)
        
    def forward(self, x):
        return self.linear(x)

# Modell, Loss-Funktion und Optimizer initialisieren
model = SimpleWordNet(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print(f"\nModell erstellt:")
print(f"Input Layer: {vocab_size} Neuronen")
print(f"Output Layer: {vocab_size} Neuronen")
print(f"\nGewichtsmatrix Form: {model.linear.weight.shape}")
print(f"Initiale Gewichte (erste 5x5):\n{model.linear.weight.data[:5, :5]}")

# Training
num_epochs = 1000
print(f"\nTraining startet für {num_epochs} Epochen...")

for epoch in range(num_epochs):
    # Forward Pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward Pass und Optimierung
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Ausgabe alle 100 Epochen
    if (epoch + 1) % 100 == 0:
        print(f'Epoche [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\nTraining abgeschlossen!")

# Test: Vorhersage des nächsten Wortes
print("\n" + "="*50)
print("TEST: Vorhersage des nächsten Wortes")
print("="*50)

test_words = vocab[:min(5, len(vocab))]  # Teste mit ersten Wörtern

for test_word in test_words:
    if test_word in word_to_idx:
        # Input vorbereiten
        input_vec = torch.zeros(vocab_size)
        input_vec[word_to_idx[test_word]] = 1.0
        
        # Vorhersage
        with torch.no_grad():
            output = model(input_vec)
            predicted_idx = int(torch.argmax(output).item())
            predicted_word = idx_to_word[predicted_idx]
            confidence = torch.softmax(output, dim=0)[predicted_idx].item()
        
        print(f"Input: '{test_word}' -> Vorhersage: '{predicted_word}' (Konfidenz: {confidence:.2%})")

print("\n" + "="*50)
print(f"Finale Gewichte (erste 5x5):\n{model.linear.weight.data[:5, :5]}")
