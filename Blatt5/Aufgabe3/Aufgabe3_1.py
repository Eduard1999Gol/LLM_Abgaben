import torch
from transformers import pipeline

# Pipeline für Text-Generierung laden
# device_map="auto" nutzt GPU falls verfügbar
evaluator = pipeline(
    "text-generation", 
    model="google/gemma-2-2b-it", 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto"
)

# Test-Sätze (Ein schlechter, ein mittelmäßiger, ein guter Yoda-Satz)
test_sentences = [
    "I will go to the market now.",             # 0% Yoda
    "To the market I will go.",                 # 50% Yoda
    "Go to the market, I will. Hmmm.",          # 100% Yoda
]

def evaluate_sentence(sentence, prompt_style="numeric"):
    
    if prompt_style == "numeric":
        # Variante A: Nur Zahl
        messages = [
            {"role": "user", "content": f"Rate the following sentence based on how much it sounds like Yoda from Star Wars on a scale from 1 to 10. Output ONLY the number.\n\nSentence: '{sentence}'"}
        ]
    elif prompt_style == "reasoning":
        # Variante B: Erklärung + Zahl
        messages = [
            {"role": "user", "content": f"Analyze the grammar and style of the following sentence. Does it use Object-Subject-Verb order? Does it use typical Yoda vocabulary? After your analysis, give a score from 1 to 10 representing 'Yoda-ness'.\n\nSentence: '{sentence}'"}
        ]
    
    # Prompt an Gemma senden
    outputs = evaluator(messages, max_new_tokens=100, do_sample=False) # do_sample=False für Deterministik
    return outputs[0]["generated_text"][-1]["content"]

print("--- 3.1 EXPERIMENTE ---")

print("\nVARIANTE A (Nur Numerisch):")
for s in test_sentences:
    print(f"Satz: '{s}' -> Score: {evaluate_sentence(s, 'numeric')}")

print("\nVARIANTE B (Mit Erklärung):")
for s in test_sentences:
    print(f"Satz: '{s}'\nAntwort: {evaluate_sentence(s, 'reasoning')}\n")

# Konsistenz-Check: Gleichen Satz 3x testen
print("\nKONSISTENZ-CHECK (Satz 2, 3x ausgeführt):")
for i in range(3):
    print(f"Versuch {i+1}: {evaluate_sentence(test_sentences[1], 'numeric')}")