import os
import glob
from bs4 import BeautifulSoup

def extract_text_from_html_files(directory_name="Datensatz", output_file="extracted_text.txt"):
    """
    Durchsucht alle HTML-Dateien in einem Unterverzeichnis des Skript-Ordners 
    und extrahiert Text aus <p class="text"> Tags.
    
    Args:
        directory_name (str): Name des Unterverzeichnisses mit HTML-Dateien (z.B. "Datensatz")
        output_file (str): Name der Ausgabedatei
    """
    extracted_texts = []
    
    # 1. Absoluten Pfad des Skript-Ordners bestimmen (z.B. /home/eduard/Desktop/LLM/Blatt1)
    # '__file__' gibt den Pfad des aktuellen Skripts zurück
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Absoluten Pfad zum Datensatz-Ordner erstellen
    data_directory_absolute = os.path.join(script_directory, directory_name)
    
    # Alle HTML-Dateien im absoluten Pfad sammeln
    html_pattern = os.path.join(data_directory_absolute, "*.html")
    html_files = glob.glob(html_pattern) 
    
    if not html_files:
        print("!!! KEINE HTML-DATEIEN GEFUNDEN. !!!")
    
    for html_file in html_files:
        print(f"Verarbeite: {os.path.basename(html_file)}")
        
        try:
            # HTML-Datei öffnen und lesen (immer den vollständigen Pfad verwenden)
            with open(html_file, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # HTML mit BeautifulSoup parsen
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Alle <p class="text"> Tags finden
            text_paragraphs = soup.find_all('p', class_="text")
            
            # Text aus den gefundenen Tags extrahieren
            for paragraph in text_paragraphs:
                text_content = paragraph.get_text(strip=True)
                if text_content:
                    extracted_texts.append(text_content)
                    
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {html_file}: {e}")
    
    # Extrahierten Text in Ausgabedatei speichern (Verwenden Sie den absoluten Skript-Pfad)
    output_path = os.path.join(script_directory, output_file)
    
    try:
        # ... (Speicherlogik bleibt gleich) ...
        with open(output_path, 'w', encoding='utf-8') as output:
            if extracted_texts:
                output.write("\n")
                output.write("=" * 50 + "\n\n")
                for text in extracted_texts:
                    output.write(text + "\n\n")
            else:
                output.write("Kein Text in <p class='text'> Tags gefunden.\n")
        
    except Exception as e:
        print(f"Fehler beim Speichern der Ausgabedatei: {e}")

# Beispielaufruf
if __name__ == "__main__":
    # Standardaufruf: Sucht im Unterordner 'Datensatz' des Skript-Verzeichnisses
    extract_text_from_html_files()