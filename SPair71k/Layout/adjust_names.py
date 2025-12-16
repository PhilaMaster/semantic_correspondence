import os
from pathlib import Path

def convert_colons_in_txt_files(root_directory):
    """
    Trova tutti i file .txt nelle sottocartelle e sostituisce ':' con '_' in ogni riga
    
    Args:
        root_directory: Directory radice dove cercare i file .txt
    """
    root_path = Path(root_directory)
    
    # Trova tutti i file .txt ricorsivamente
    txt_files = list(root_path.rglob('*.txt'))
    
    if not txt_files:
        print(f"Nessun file .txt trovato in {root_directory}")
        return
    
    print(f"Trovati {len(txt_files)} file .txt da processare...\n")
    
    files_modified = 0
    total_lines_modified = 0
    
    for txt_file in txt_files:
        try:
            # Leggi il contenuto originale
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Conta quante righe contengono ':'
            lines_with_colon = sum(1 for line in lines if ':' in line)
            
            if lines_with_colon == 0:
                print(f"⊘ {txt_file.relative_to(root_path)} - Nessuna modifica necessaria")
                continue
            
            # Sostituisci ':' con '_' in ogni riga
            modified_lines = [line.replace(':', '_') for line in lines]
            
            # Scrivi il file modificato
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
            
            files_modified += 1
            total_lines_modified += lines_with_colon
            
            print(f"✓ {txt_file.relative_to(root_path)} - {lines_with_colon} righe modificate")
            
        except Exception as e:
            print(f"✗ Errore processando {txt_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"RIEPILOGO:")
    print(f"  File modificati: {files_modified}/{len(txt_files)}")
    print(f"  Righe totali modificate: {total_lines_modified}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Specifica la directory radice
    root_dir = '.'  # Directory corrente, oppure specifica il path
    
    # Opzionale: chiedi conferma prima di procedere
    print(f"Questo script modificherà tutti i file .txt in '{root_dir}' e sottocartelle")
    print(f"Sostituendo ':' con '_' in ogni riga.\n")
    
    risposta = input("Vuoi procedere? (s/n): ").strip().lower()
    
    if risposta in ['s', 'si', 'y', 'yes']:
        convert_colons_in_txt_files(root_dir)
    else:
        print("Operazione annullata.")
