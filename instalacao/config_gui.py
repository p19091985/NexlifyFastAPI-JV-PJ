import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import shutil
from pathlib import Path
import configparser

try:
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()

    config_file_path = project_root / "config_settings.ini"

    if not config_file_path.is_file():
        try:
            default_ini_content = """[Settings]
database_enabled = True
initialize_database_on_startup = True
use_login = True
redirect_console_to_log = False
"""
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(default_ini_content)
        except Exception as e:

            raise FileNotFoundError(f"Não foi possível criar {config_file_path}: {e}")

except Exception as e:

    messagebox.showerror("Erro Crítico",
                         f"Erro ao localizar/criar 'config_settings.ini' na pasta raiz do projeto: {e}")
    sys.exit(1)

class ConfigApp(tk.Tk):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

        self.parser = configparser.ConfigParser()

        self.title("Configurador Inteligente (config_settings.ini)")

        w = 600
        h = 550                   
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw // 2) - (w // 2)
        y = (sh // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(550, 550)                          

        self.db_enabled_var = tk.BooleanVar()
        self.init_db_var = tk.BooleanVar()
        self.use_login_var = tk.BooleanVar()
        self.redirect_log_var = tk.BooleanVar()

        self.vars_map = {
            "database_enabled": self.db_enabled_var,
            "initialize_database_on_startup": self.init_db_var,
            "use_login": self.use_login_var,
            "redirect_console_to_log": self.redirect_log_var,
        }

        self.original_save_button_text = "Salvar Configurações"

        self._setup_styles()
        self._create_widgets()
        self._load_initial_values()
        self._on_db_setting_change()

    def _setup_styles(self):
        """Configura estilos visuais (Idêntico à GUI original da Streamlit)."""
        style = ttk.Style(self)
        style.configure("Success.TButton", foreground="white", background="#28a745", font=("-weight", "bold"))
        style.map("Success.TButton", background=[('active', '#218838')])
        style.configure("Success.TLabel", foreground="#28a745", font=("-size", 9, "-weight", "bold"))
        style.configure("Warning.TLabel", foreground="#ff8c00", font=("-size", 9, "-weight", "bold"))
        style.configure("TLabel", font=("-size", 10))
        style.configure("Help.TLabel", font=("-size", 9), foreground="#555")
        style.configure("Header.TLabel", font=("-size", 14, "-weight", "bold"))
        style.configure("TLabelframe.Label", font=("-size", 11, "-weight", "bold"), foreground="#005a9e")

    def _create_widgets(self):
        """Cria a interface do usuário (Adaptado para incluir a configuração de tema)."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Configurações Principais (config_settings.ini)", style="Header.TLabel").pack(
            pady=(0, 15), anchor="w")

        db_frame = ttk.LabelFrame(main_frame, text="🎲 Conexão Principal", padding=15)

        db_frame.pack(fill=tk.X, expand=False, pady=5)
        self.db_enabled_check = ttk.Checkbutton(db_frame,

                                                text="Habilitar Banco de Dados (database_enabled)",
                                                variable=self.db_enabled_var,
                                                command=self._on_db_setting_change)
        self.db_enabled_check.pack(anchor="w")
        ttk.Label(db_frame,
                  text="Controla se a aplicação se conecta a um banco de dados.",
                  style="Help.TLabel", wraplength=500).pack(anchor="w", padx=20, pady=(0, 5))

        dependent_frame = ttk.LabelFrame(main_frame, text="⚙️ Módulos Dependentes", padding=15)

        dependent_frame.pack(fill=tk.X, expand=False, pady=10)
        self.dependency_status_label = ttk.Label(dependent_frame, text="", style="Warning.TLabel", justify="center")
        self.dependency_status_label.pack(fill='x', pady=(0, 10))

        self.init_db_check = ttk.Checkbutton(dependent_frame,

                                             text="Inicializar Banco ao Iniciar (initialize_database_on_startup)",
                                             variable=self.init_db_var)
        self.init_db_check.pack(anchor="w", pady=(5, 0))
        ttk.Label(dependent_frame,
                  text="Se ativo, cria o schema do banco (SQLite). Depende do DB Habilitado.",
                  style="Help.TLabel", wraplength=500).pack(anchor="w", padx=20, pady=(0, 5))

        self.use_login_check = ttk.Checkbutton(dependent_frame,

                                               text="Exigir Login de Usuário (use_login)",
                                               variable=self.use_login_var)
        self.use_login_check.pack(anchor="w", pady=(5, 0))
        ttk.Label(dependent_frame,
                  text="Se ativo, exige autenticação. Depende do DB Habilitado.",
                  style="Help.TLabel", wraplength=500).pack(anchor="w", padx=20, pady=(0, 5))

        general_frame = ttk.LabelFrame(main_frame, text="🔧 Configurações Gerais", padding=15)

        general_frame.pack(fill=tk.X, expand=False, pady=5)

        self.redirect_log_check = ttk.Checkbutton(general_frame,

                                                  text="Redirecionar Console para Log (redirect_console_to_log)",
                                                  variable=self.redirect_log_var)
        self.redirect_log_check.pack(anchor="w")
        ttk.Label(general_frame,
                  text="Se ativo, a saída do console (print) vai para arquivos de log.",
                  style="Help.TLabel", wraplength=500).pack(anchor="w", padx=20, pady=(0, 5))

        self.save_button = ttk.Button(main_frame, text=self.original_save_button_text, command=self._save_settings,
                                      style="Success.TButton")
        self.save_button.pack(pady=(20, 5), fill=tk.X, ipady=5)
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.pack(pady=(5, 0))

    def _on_db_setting_change(self, *args):
        """Atualiza estado dos checkbuttons dependentes (Mesma lógica)."""
        is_db_enabled = self.db_enabled_var.get()
        if is_db_enabled:
            self.init_db_check.config(state="normal")
            self.use_login_check.config(state="normal")
            self.dependency_status_label.config(
                text="Banco HABILITADO. Opções dependentes disponíveis.",
                style="Success.TLabel"
            )
        else:
            self.init_db_check.config(state="disabled")
            self.use_login_check.config(state="disabled")

            self.init_db_var.set(False)
            self.use_login_var.set(False)
            self.dependency_status_label.config(
                text="Banco DESABILITADO. Login e Inicialização foram desativados.",
                style="Warning.TLabel"
            )

    def _load_initial_values(self):
        """Lê o config_settings.ini usando configparser."""
        try:

            self.parser.read(self.config_path)

            if 'Settings' not in self.parser:
                self.parser['Settings'] = {}
                self._update_status("Aviso: Seção [Settings] não encontrada no .ini. Usando padrões.", "orange")

            for key, tk_var in self.vars_map.items():
                value = self.parser.getboolean('Settings', key, fallback=False)
                tk_var.set(value)

        except configparser.Error as e:
            messagebox.showerror("Erro de Leitura INI", f"Erro ao processar '{self.config_path}':\n{e}", parent=self)
            self.save_button.config(state="disabled")
            self._update_status(f"Erro ao ler .ini: {e}", "red")
        except Exception as e:
            messagebox.showerror("Erro ao Ler", f"Não foi possível ler '{self.config_path}':\n{e}", parent=self)
            self.save_button.config(state="disabled")
            self._update_status(f"Erro geral ao ler .ini: {e}", "red")

        self._on_db_setting_change()

    def _save_settings(self):
        """Salva as configurações no config_settings.ini usando configparser."""

        self.save_button.config(state="disabled", text="Salvando...")
        self._update_status("Salvando configurações...", "blue")
        self.update_idletasks()

        try:

            backup_path = self.config_path.with_suffix(".ini.bak")
            shutil.copy2(self.config_path, backup_path)

            if 'Settings' not in self.parser:
                self.parser['Settings'] = {}

            for key, tk_var in self.vars_map.items():
                self.parser.set('Settings', key, str(tk_var.get()))

            with open(self.config_path, 'w', encoding='utf-8') as configfile:
                self.parser.write(configfile)

            self._update_status(f"Salvo! Backup: {backup_path.name}", "green")
            self.save_button.config(text="✔ Salvo com Sucesso!")

            self.after(2000, self._revert_save_button)

        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar '{self.config_path}':\n{e}", parent=self)
            self._update_status(f"Erro ao salvar: {e}", "red")

            self._revert_save_button()

    def _revert_save_button(self):
        """Restaura o botão de salvar ao seu estado original."""
        try:

            if self.winfo_exists():
                self.save_button.config(state="normal", text=self.original_save_button_text)
        except tk.TclError:

            pass

    def _update_status(self, message, color):
        """Atualiza o label de status principal."""
        self.status_label.config(text=message, foreground=color)

if __name__ == "__main__":
    app = ConfigApp(config_file_path)
    app.mainloop()