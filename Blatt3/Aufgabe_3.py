from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def start_chat_app():
    # --- Setup ---
    model_id = "unsloth/gemma-2-2b-it"
    print("Initialisiere Chatbot... Bitte warten.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print("Bereit! (Schreibe 'exit' zum Beenden)")

    # --- Die Chat-Schleife ---
    while True:
        # (b) Eingabe aus der Konsole lesen
        user_input = input("\nDeine Frage: ")
        
        if user_input.lower() in ["exit", "quit", "ende"]:
            print("Programm beendet.")
            break
            
        # (a) Verschiedene Fragen in den Prompt einfügen (via f-string)
        # Wir bauen den Prompt dynamisch zusammen
        prompt_template = f"<bos><start_of_turn>user\n{user_input}<eSinus-/ Cosinus- Funktionennd_of_turn>\n<start_of_turn>model"
        
        # Generierungsprozess
        input_ids = tokenizer(prompt_template, return_tensors="pt").to(model.device)
        
        # Hier generieren wir die Antwort
        outputs = model.generate(**input_ids, max_new_tokens=100)
        
        # Wir dekodieren die ALLES, aber wir wollen eigentlich nur den neuen Teil anzeigen.
        # Ein einfacher Trick ist, die Länge des Inputs abzuziehen oder direkt zu decodieren.
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Um nur die Antwort des Modells sauber zu extrahieren (ohne den Prompt zu wiederholen):
        # Wir suchen nach dem Start der Modell-Antwort im String oder nutzen slicing beim decoding.
        # Einfachste Variante hier:
        model_answer = full_response.replace(user_input, "").replace("user", "").replace("model", "").strip()

        print(f"Gemma: {model_answer}")

if __name__ == "__main__":
    start_chat_app()