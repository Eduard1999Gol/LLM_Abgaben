from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from Aufgabe2_1 import model, tokenizer, device, prompts
from Aufgabe2_2 import tokenized_datasets

# 1. LoRA Konfiguration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # Rank der Matrizen (kleiner = weniger Parameter, aber weniger Kapazität)
    lora_alpha=32,  # Skalierungsfaktor
    lora_dropout=0.1,
    # Bei GPT-2 heißt das Attention-Modul oft 'c_attn'
    target_modules=["c_attn"] 
)

# 2. Modell vorbereiten
model_peft = get_peft_model(model, peft_config)
model_peft.print_trainable_parameters() 
# Ausgabe zeigt, dass wir nur sehr wenige Parameter trainieren (oft < 1%)

# 3. Training Argumente
training_args = TrainingArguments(
    output_dir="./yoda-distilgpt2-lora",
    per_device_train_batch_size=4,
    num_train_epochs=10, # Bei so wenigen Daten brauchen wir viele Epochen (im echten Set reichen oft 3-5)
    learning_rate=5e-4,  # LoRA verträgt oft höhere Learning Rates
    logging_steps=1,
    save_strategy="no"   # Sparen wir uns Speicherplatz für dieses Beispiel
)

# 4. Trainer initialisieren
trainer = Trainer(
    model=model_peft,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 5. Training starten
print("\n--- Starte Training ---")
trainer.train()

# --- Nach dem Training: Ergebnisse prüfen ---
# Wir nutzen das trainierte Modell für dieselben Prompts wie in 2.1

model_peft.eval() # In den Evaluierungsmodus wechseln

print("\n--- 2.3 Ausgaben NACH dem Fine-Tuning (LoRA) ---")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model_peft.generate(
        **inputs, 
        max_new_tokens=15, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extrahiere nur die neue Antwort
    answer = result.split('Yoda:')[-1].strip()
    print(f"\nPrompt: {prompt.splitlines()[0]}\nYoda: {answer}")