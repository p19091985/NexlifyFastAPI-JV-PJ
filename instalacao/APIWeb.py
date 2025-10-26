import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import urllib.request
import threading
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

FILES_TO_DOWNLOAD = {
    "static/css/bootstrap.min.css": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
    "static/js/bootstrap.bundle.min.js": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
    "static/js/htmx.min.js": "https://cdn.jsdelivr.net/npm/htmx.org@2.0.0/dist/htmx.min.js",
    "static/js/alpine.min.js": "https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"
}

class InstallerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Instalador de Dependências Frontend")
        self.root.geometry("650x450")
        self.root.resizable(False, False)

        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')

        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)

        title_label = ttk.Label(
            main_frame,
            text="Instalador (Bootstrap, HTMX, Alpine.js)",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        desc_text = (
            f"Este script será executado a partir de: \n{SCRIPT_DIR}\n\n"
            f"Ele irá baixar e instalar os arquivos necessários em: \n{BASE_DIR / 'static'}"
        )
        desc_label = ttk.Label(main_frame, text=desc_text, justify="left")
        desc_label.pack(pady=5, fill="x")

        self.start_button = ttk.Button(
            main_frame,
            text="Iniciar Instalação",
            command=self.start_installation_thread
        )
        self.start_button.pack(pady=15, fill="x", ipady=5)

        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill="both", expand=True)

        log_label = ttk.Label(log_frame, text="Log de Atividades:")
        log_label.pack(anchor="w")

        self.log_area = ScrolledText(
            log_frame,
            height=12,
            state="disabled",
            wrap=tk.WORD,
            font=("Courier New", 9)
        )
        self.log_area.pack(fill="both", expand=True, pady=(5, 0))

    def log_message(self, message):
        """ Adiciona uma mensagem à caixa de log (thread-safe) """
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def start_installation_thread(self):
        """ Inicia a instalação em uma thread separada para não congelar a GUI """
        self.start_button.config(state="disabled", text="Instalando...")
        self.log_area.config(state="normal")
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state="disabled")

        install_thread = threading.Thread(target=self.run_installation, daemon=True)
        install_thread.start()

    def run_installation(self):
        """ O processo real de download e salvamento de arquivos """
        try:
            self.log_message("--- Iniciando instalação ---")

            for relative_path, url in FILES_TO_DOWNLOAD.items():
                target_path = BASE_DIR / relative_path
                target_dir = target_path.parent

                self.log_message(f"\nProcessando: {relative_path}")

                if not target_dir.exists():
                    self.log_message(f"  Criando diretório: {target_dir}")
                    os.makedirs(target_dir)
                else:
                    self.log_message(f"  Diretório já existe.")

                self.log_message(f"  Baixando de: {url}")
                with urllib.request.urlopen(url) as response:
                    if response.status != 200:
                        raise Exception(f"Falha ao baixar (Status: {response.status})")
                    content = response.read()

                with open(target_path, 'wb') as f:
                    f.write(content)
                self.log_message(f"  Arquivo salvo com sucesso ({len(content)} bytes).")

            self.log_message("\n----------------------------------")
            self.log_message("TUDO OK! Instalação concluída com sucesso.")

            messagebox.showinfo(
                "Instalação Concluída",
                "Tudo OK!\n\nOs arquivos estáticos (Bootstrap, HTMX, Alpine.js) foram baixados e instalados com sucesso."
            )

        except Exception as e:
            error_msg = f"\nERRO: {e}"
            self.log_message(error_msg)
            messagebox.showerror("Erro na Instalação", f"Ocorreu um erro:\n{e}")
        finally:
            self.start_button.config(state="normal", text="Iniciar Instalação")

if __name__ == "__main__":
    root = tk.Tk()
    app = InstallerApp(root)
    root.mainloop()