import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import hashlib
from pathlib import Path
import platform

SCRIPT_DIR = Path(__file__).parent
ENV_FILE_PATH = SCRIPT_DIR.parent / ".env"                             
                                                               
SALT = b"nexlify_fastapi_salt_!@#$%^"

def generate_key_from_words(words_string):
    """Gera uma chave SHA-256 a partir de uma string de palavras."""
    if not words_string.strip():
        return None

    salted_words = SALT + words_string.encode('utf-8')

    hasher = hashlib.sha256()
    hasher.update(salted_words)

    return hasher.hexdigest()

def write_env_file(secret_key):
    """Escreve ou atualiza a SESSION_SECRET_KEY no arquivo .env."""
    env_content = ""
    key_found = False

    try:
                                        
        if ENV_FILE_PATH.exists():
            with open(ENV_FILE_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                if line.strip().startswith("SESSION_SECRET_KEY="):
                    env_content += f'SESSION_SECRET_KEY="{secret_key}"\n'
                    key_found = True
                else:
                    env_content += line

        if not key_found:
            env_content += f'\nSESSION_SECRET_KEY="{secret_key}"\n'

        env_content = env_content.strip() + "\n"

        with open(ENV_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(env_content)
        return True

    except IOError as e:
        messagebox.showerror("Erro de Arquivo", f"Não foi possível escrever no arquivo {ENV_FILE_PATH}:\n{e}")
        return False
    except Exception as e:
        messagebox.showerror("Erro Inesperado", f"Ocorreu um erro ao processar o arquivo .env:\n{e}")
        return False

def handle_generate_and_write():
    """Função chamada pelo botão."""
    input_words = words_entry.get()

    generated_key = generate_key_from_words(input_words)

    if generated_key:
                                                         
        key_display_text.config(state=tk.NORMAL)
        key_display_text.delete('1.0', tk.END)
        key_display_text.insert(tk.END, generated_key)
        key_display_text.config(state=tk.DISABLED)

        if write_env_file(generated_key):
            status_label.config(text=f"Sucesso! Chave gerada e salva em {ENV_FILE_PATH.name}")
            messagebox.showinfo("Sucesso",
                                f"A chave foi gerada com sucesso e salva no arquivo '{ENV_FILE_PATH.name}' na pasta raiz do projeto.")
        else:
            status_label.config(text=f"Erro ao escrever no arquivo {ENV_FILE_PATH.name}")

    else:
                                   
        key_display_text.config(state=tk.NORMAL)
        key_display_text.delete('1.0', tk.END)
        key_display_text.config(state=tk.DISABLED)

        status_label.config(text="Erro: Por favor, insira algumas palavras.")
        messagebox.showwarning("Entrada Vazia", "Por favor, digite algumas palavras ou frases para gerar a chave.")

root = tk.Tk()
root.title("Gerador de Chave Secreta para .env")
root.geometry("600x400")
root.resizable(False, False)

style = ttk.Style(root)
style.theme_use('clam')

default_font = ("TkDefaultFont", 10)
if platform.system() == "Windows":
    default_font = ("Segoe UI", 10)
elif platform.system() == "Darwin":         
    default_font = ("Helvetica", 11)

main_frame = ttk.Frame(root, padding="15 15 15 15")
main_frame.pack(fill=tk.BOTH, expand=True)

instruction_label = ttk.Label(
    main_frame,
    text="Digite algumas palavras ou frases aleatórias abaixo.\nElas serão usadas para gerar uma chave secreta segura (SHA-256).",
    justify=tk.CENTER,
    font=(default_font[0], 11)
)
instruction_label.pack(pady=(0, 10))

words_entry = ttk.Entry(main_frame, width=60, font=default_font)
words_entry.pack(pady=5, ipady=4)
words_entry.focus()

generate_button = ttk.Button(
    main_frame,
    text="Gerar Chave e Salvar no .env",
    command=handle_generate_and_write,
    style='Accent.TButton'                                 
)
style.configure('Accent.TButton', font=(default_font[0], 10, 'bold'))
generate_button.pack(pady=15, fill=tk.X, ipady=5)

key_label = ttk.Label(main_frame, text="Chave Gerada:")
key_label.pack(anchor=tk.W, pady=(10, 2))

key_display_text = scrolledtext.ScrolledText(
    main_frame,
    height=4,
    wrap=tk.WORD,
    state=tk.DISABLED,
    font=("Courier New", 10),
    relief=tk.SUNKEN,
    borderwidth=1
)
key_display_text.pack(fill=tk.X, expand=False, pady=(0, 10))

status_label = ttk.Label(main_frame, text="Aguardando entrada...", relief=tk.SUNKEN, anchor=tk.W, padding=5)
status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

root.mainloop()