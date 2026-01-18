import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

# Importiere deine Vorbereitung
from dataset_vorbereitung import TextDataset, create_train_test_split

# --- AUFGABE 3: DAS MODELL ---
class BengioLanguageModel(nn.Module):
    """
    Unser Sprachmodell basierend auf Bengio et al. (2003).
    Architektur: Embedding -> Flatten -> Hidden (Tanh) -> Output (LogSoftmax)
    """
    def __init__(self, vocab_size, embedding_dim=30, context_size=3, hidden_size=100):
        super(BengioLanguageModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        
        # 1. Embedding Layer (C)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Hidden Layer (H)
        # Input Dimension: 3 Wörter * 30 Dim = 90
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        
        # 3. Output Layer (O)
        # Input Dimension: 100 Hidden Units -> Output: Vokabulargröße
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        # inputs shape: (batch_size, context_size) -> z.B. (32, 3)
        
        # 1. Lookup
        embeds = self.embeddings(inputs) 
        # shape: (batch_size, 3, 30)
        
        # 2. Flatten / Concatenate
        embeds = embeds.view(-1, self.context_size * self.embedding_dim)
        # shape: (batch_size, 90)
        
        # 3. Hidden Layer + Tanh Activation
        out = torch.tanh(self.linear1(embeds))
        # shape: (batch_size, 100)
        
        # 4. Output Layer
        out = self.linear2(out)
        # shape: (batch_size, vocab_size)
        
        # 5. Log Softmax (für NLL Loss)
        log_probs = F.log_softmax(out, dim=1)
        
        return log_probs

# --- HILFSFUNKTIONEN FÜR TRAINING ---

def train_one_epoch(model, train_loader, optimizer, criterion, epoch_index):
    model.train() # Setzt das Modell in den Trainings-Modus (wichtig für Dropout etc.)
    running_loss = 0.0
    
    for i, (context_batch, target_batch) in enumerate(train_loader):
        # 1. Gradients nullen (sonst summieren sie sich auf)
        optimizer.zero_grad()
        
        # 2. Forward Pass (Vorhersage berechnen)
        log_probs = model(context_batch)
        
        # 3. Loss berechnen (Vergleich Vorhersage vs. Ziel)
        loss = criterion(log_probs, target_batch)
        
        # 4. Backward Pass (Fehler zurückrechnen)
        loss.backward()
        
        # 5. Optimizer Step (Gewichte anpassen)
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate(model, test_loader, criterion):
    model.eval() # Setzt das Modell in den Evaluierungs-Modus
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # torch.no_grad() spart Speicher, da wir hier keine Gradients brauchen
    with torch.no_grad():
        for context_batch, target_batch in test_loader:
            log_probs = model(context_batch)
            loss = criterion(log_probs, target_batch)
            running_loss += loss.item()
            
            # Genauigkeit berechnen (optional, aber interessant)
            # Wir nehmen den Index mit der höchsten Wahrscheinlichkeit (argmax)
            predictions = torch.argmax(log_probs, dim=1)
            correct_predictions += (predictions == target_batch).sum().item()
            total_predictions += target_batch.size(0)
            
    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


# --- AUFGABE 4: HAUPTPROGRAMM (TRAINING) ---
if __name__ == "__main__":
    
    # 1. Daten laden
    print("Lade Dataset...")
    full_dataset = TextDataset('text.txt', context_size=3)
    train_loader, test_loader, _ = create_train_test_split(full_dataset, batch_size=32)
    vocab_size = full_dataset.vocab_size

    # 2. Modell initialisieren
    print(f"\nInitialisiere Modell für Vokabulargröße {vocab_size}...")
    model = BengioLanguageModel(vocab_size=vocab_size, embedding_dim=30, context_size=3, hidden_size=100)
    
    # 3. Loss Function & Optimizer definieren
    # NLLLoss erwartet Log-Wahrscheinlichkeiten (die unser Modell liefert)
    criterion = nn.NLLLoss()
    
    # Adam ist meist robuster als SGD. Learning Rate 0.001 ist Standard.
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Training Loop
    EPOCHS = 10  # Wie oft gehen wir durch den ganzen Text?
    
    print(f"\nStarte Training für {EPOCHS} Epochen...")
    print("-" * 60)
    print(f"{'Epoche':<10} | {'Train Loss':<15} | {'Test Loss':<15} | {'Test Acc':<10} | {'Zeit'}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Trainieren
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # Evaluieren (Testen)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        epoch_duration = time.time() - epoch_start
        
        print(f"{epoch+1:<10} | {train_loss:.4f}          | {test_loss:.4f}          | {test_acc*100:.1f}%      | {epoch_duration:.1f}s")

    total_time = time.time() - start_time
    print("-" * 60)
    print("Training abgeschlossen!")
    
    # 5. Finales Ergebnis
    final_test_loss, final_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinaler Loss auf dem Test-Datensatz: {final_test_loss:.4f}")
    print(f"Perplexity (Unsicherheit): {torch.exp(torch.tensor(final_test_loss)):.2f}")
    
    # Kleiner Test der Vorhersage
    print("\nBeispiel-Vorhersage (zufällig aus Testdaten):")
    model.eval()
    context, target = next(iter(test_loader))
    with torch.no_grad():
        output = model(context)
        pred_idx = torch.argmax(output[0]).item()
    
    # Konvertiere Indizes zu Wörtern
    context_words = [full_dataset.idx_to_word[idx.item()] for idx in context[0]]
    predicted_word = full_dataset.idx_to_word[pred_idx]
    target_word = full_dataset.idx_to_word[target[0].item()]
    
    print(f"Input Wörter: {context_words}")
    print(f"Vorhergesagtes Wort: '{predicted_word}' (Index: {pred_idx})")
    print(f"Echtes Wort: '{target_word}' (Index: {target[0].item()})")
    print(f"Korrekt: {'✓' if pred_idx == target[0].item() else '✗'}")