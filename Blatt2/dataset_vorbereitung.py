## Aufgabe 2: Dataset-Vorbereitung für 3-Wort-Kontext Sprachmodell

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import re
from collections import Counter

class TextDataset(Dataset):
    """Dataset für 3-Wort-Kontext Sprachmodell"""
    
    def __init__(self, text_file, context_size=3):
        """
        Args:
            text_file: Pfad zur Textdatei
            context_size: Anzahl der Wörter im Kontext (Standard: 3)
        """
        self.context_size = context_size
        
        # Text einlesen und vorverarbeiten
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Text in Wörter tokenisieren (einfache Tokenisierung)
        words = self._tokenize(text)
        print(f"Gesamtanzahl Wörter: {len(words)}")
        
        # Vokabular erstellen
        self.word_counts = Counter(words)
        self.vocab = ['<UNK>'] + sorted(self.word_counts.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vokabulargröße: {self.vocab_size}")
        
        # Sequenzen erstellen (x: 3 Wörter, y: nächstes Wort)
        self.sequences = []
        for i in range(len(words) - context_size):
            context = words[i:i+context_size]   # 3 Wörter Kontext
            target = words[i+context_size]      # Nächstes Wort
            
            # In Indizes umwandeln
            context_idx = [self.word_to_idx.get(w, 0) for w in context]
            target_idx = self.word_to_idx.get(target, 0)
            
            self.sequences.append((context_idx, target_idx))
        
        print(f"Anzahl Trainingssequenzen: {len(self.sequences)}")
    
    def _tokenize(self, text):
        """Einfache Tokenisierung in Wörter"""
        # Text in Kleinbuchstaben umwandeln
        text = text.lower()
        # Nur Buchstaben, Zahlen und Leerzeichen behalten
        text = re.sub(r'[^\w\s]', ' ', text)
        # In Wörter aufteilen
        words = text.split()
        # Leere Strings entfernen
        words = [w for w in words if w]
        return words
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        context, target = self.sequences[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def create_train_test_split(dataset, train_ratio=0.8, batch_size=32):
    """
    Teilt den Datensatz in Training und Test auf
    
    Args:
        dataset: Das komplette Dataset
        train_ratio: Anteil der Trainingsdaten (Standard: 0.8)
        batch_size: Batch-Größe für DataLoader
    
    Returns:
        train_loader, test_loader, dataset
    """
    # Größen berechnen
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    print(f"\nDataset-Split:")
    print(f"Training: {train_size} Sequenzen ({train_ratio*100:.0f}%)")
    print(f"Test: {test_size} Sequenzen ({(1-train_ratio)*100:.0f}%)")
    
    # Split durchführen
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Für Reproduzierbarkeit
    )
    
    # DataLoader erstellen
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, dataset


if __name__ == "__main__":
    # Dataset erstellen
    print("=" * 60)
    print("Aufgabe 1: Dataset mit 3-Wort-Kontext erstellen")
    print("=" * 60)
    
    dataset = TextDataset('text.txt', context_size=3)
    
    # Beispiele anzeigen
    print("\nBeispiel-Sequenzen:")
    for i in range(3):
        context, target = dataset[i]
        context_words = [dataset.idx_to_word[idx.item()] for idx in context]
        target_word = dataset.idx_to_word[target.item()]
        print(f"  Kontext: {context_words} -> Ziel: {target_word}")
    
    # Aufgabe 2: Train-Test-Split
    print("\n" + "=" * 60)
    print("Aufgabe 2: Train-Test-Split (80/20)")
    print("=" * 60)
    
    train_loader, test_loader, dataset = create_train_test_split(
        dataset, 
        train_ratio=0.8, 
        batch_size=32
    )
    
    # Beispiel-Batch anzeigen
    print("\nBeispiel-Batch aus Training:")
    for context_batch, target_batch in train_loader:
        print(f"  Context Shape: {context_batch.shape}")  # [batch_size, 3]
        print(f"  Target Shape: {target_batch.shape}")    # [batch_size]
        print(f"  Erstes Beispiel im Batch:")
        context_words = [dataset.idx_to_word[idx.item()] for idx in context_batch[0]]
        target_word = dataset.idx_to_word[target_batch[0].item()]
        print(f"    Kontext: {context_words} -> Ziel: {target_word}")
        break
    
    print("\n" + "=" * 60)
    print("Dataset-Vorbereitung abgeschlossen!")
    print("=" * 60)
    print(f"\nZusammenfassung:")
    print(f"  - Vokabulargröße: {dataset.vocab_size}")
    print(f"  - Kontext-Größe: 3 Wörter")
    print(f"  - Training Batches: {len(train_loader)}")
    print(f"  - Test Batches: {len(test_loader)}")
