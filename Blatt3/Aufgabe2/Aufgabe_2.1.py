from transformers import AutoTokenizer

# 1. Die Sätze (Edge Cases für Normalisierung)
sentences = [
    "Das ist ein einfacher Satz.",                  # Standard
    "HuggingFace ist c00l!",                        # Alphanumerisch + Slang
    "Ich wohne in der Mühlenstraße 17.",            # Umlaute + Zahlen
    "  Zu viele   Leerzeichen  hier.  ",            # Whitespace Normalisierung
    "HELLO WORLD (UPPERCASE)",                      # Casing
    "Er sagte: 'Hallo!' - oder?",                   # Interpunktion
    "test_variable_name_snake_case",                # Code-ähnlich
    "Der Preis beträgt 19.99$.",                    # Währung/Symbole
    "Ein sehr langeszusammengesetzteswort.",        # Unbekannte Token
    "Emoji Test: 😊🚀"                               # Unicode/Emojis
]

# 2. Modelle laden
model_names = ["bert-base-uncased", "gpt2", "t5-small"]

print(f"{'Original':<40} | {'Model':<10} |Tokens")
print("-" * 80)

for sent in sentences[-2:]:
    print(f"Input: {sent}")
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Tokenizer-Output holen
        tokens = tokenizer.tokenize(sent)
        print(f"{'':<40} | {model_name:<10} | {tokens}")
    print("-" * 80)


# gpt2 ist besser.

# Warum? Da du einen Transformer für Textgenerierung (ähnlich wie TinyStories) bauen willst, ist GPT-2 der Standard.
# Decoder-Only Architektur: GPT-2 ist genau dafür gemacht, das nächste Wort vorherzusagen.
# No Case Folding: Für Geschichten ist es wichtig, dass "Sie" (Person) und "sie" (Pronomen) unterschieden werden können. BERTs "uncased" Ansatz wäre hier fatal für die Qualität der Geschichte.
# Byte-Level BPE: Es ist sehr robust und produziert selten [UNK] (Unknown Token).    