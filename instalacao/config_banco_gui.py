import tkinter as tk
from tkinter import ttk, messagebox             
import os
import re
from pathlib import Path

BLOCK_DEFINITIONS = {  
    'sqlserver': ['type', 'host', 'port', 'dbname', 'user', 'password'],  
    'sqlite': ['type', 'path'],  
    'postgresql': ['type', 'host', 'port', 'dbname', 'user', 'password'],  
    'mysql': ['type', 'host', 'port', 'dbname', 'user', 'password'],  
    'mariadb': ['type', 'host', 'port', 'dbname', 'user', 'password']   
}

class DatabaseConfigManager:  
    BLOCK_DEFINITIONS = BLOCK_DEFINITIONS                                 

    def __init__(self, filepath='banco.ini'):
        self.filepath = Path(filepath)            
        self.lines = []
        self.blocks = []
        self.current_active_block_name = None

    def load_config(self):
        if not self.filepath.is_file():                      
            raise FileNotFoundError(   
                f"Arquivo não encontrado: {self.filepath}\nVerifique se 'banco.ini' está na pasta raiz do projeto.")   

        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.blocks = []
        self.current_active_block_name = None
        type_regex = re.compile(r"^\s*(#)?\s*type\s*=\s*(\w+)")

        key_val_regex = re.compile(r"^\s*(#)?\s*(\w+)\s*=")

        block_ranges = []
        current_block_start = -1
        current_db_type = None
        current_is_commented = True

        for i, line in enumerate(self.lines):   
            type_match = type_regex.match(line)
            if type_match:
                                                                           
                if current_block_start != -1:
                    block_ranges.append({
                        'type': current_db_type,
                        'start': current_block_start,
                        'end': i,
                        'is_commented': current_is_commented,
                        'line_num': current_block_start + 1
                    })

                current_is_commented = type_match.group(1) is not None
                current_db_type = type_match.group(2)
                current_block_start = i
                                                                         
        if current_block_start != -1:
             block_ranges.append({
                'type': current_db_type,
                'start': current_block_start,
                'end': len(self.lines),
                'is_commented': current_is_commented,
                'line_num': current_block_start + 1
            })

        for block_info in block_ranges:
            db_type = block_info['type']
            if db_type in self.BLOCK_DEFINITIONS:   
                expected_keys = self.BLOCK_DEFINITIONS[db_type]   
                line_indices = {}

                for line_index in range(block_info['start'], block_info['end']):
                    line = self.lines[line_index].strip()
                    key_match = key_val_regex.match(line)
                    if key_match:
                        key_found = key_match.group(2)
                                                                                              
                        if key_found in expected_keys and key_found not in line_indices:   
                            line_indices[key_found] = line_index

                block_name = f"{db_type} (Linha {block_info['line_num']})"
                self.blocks.append({'name': block_name, 'indices_dict': line_indices})                  

                if not block_info['is_commented']:
                    self.current_active_block_name = block_name
            else:
                 print(f"Aviso: Tipo de banco desconhecido '{db_type}' na linha {block_info['line_num']} ignorado.")

    def get_all_dbs(self):
        return [block['name'] for block in self.blocks]

    def get_active_db(self):
        return self.current_active_block_name

    def activate_db(self, name_to_activate):   
                                           
        for block in self.blocks:
            is_target_block = (block['name'] == name_to_activate)
                                                                                
            for key, line_index in block['indices_dict'].items():             
                                                                                 
                original_line = self.lines[line_index]
                stripped_line = original_line.lstrip()
                indent_space = original_line[:len(original_line) - len(stripped_line)]
                content = stripped_line.lstrip('#').lstrip()   

                if is_target_block:
                                                
                    new_line = f"{indent_space}{content}"  
                else:
                                                
                    if not content.startswith('#'):                     
                       new_line = f"{indent_space}#{content}"  
                    else:
                       new_line = f"{indent_space}{content}"

                self.lines[line_index] = new_line.rstrip() + os.linesep

        self.current_active_block_name = name_to_activate

    def save_config(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.writelines(self.lines)

class App(tk.Tk):
    def __init__(self, manager: DatabaseConfigManager):
        super().__init__()
        self.manager = manager   

        self.title("Configurador de Conexão (banco.ini)")                    
        self.geometry("500x400")                     
        self.minsize(450, 439)

        self._setup_styles()

        self.selected_db = tk.StringVar()  
                                        
        self.status_label = ttk.Label(self, text="", style="Status.TLabel")

        main_container = ttk.Frame(self, padding=15)   
        main_container.pack(fill="both", expand=True)   

        self.create_widgets(main_container)                    
        self._load_and_populate()                                          

    def _setup_styles(self):
        """Configura estilos ttk."""
        style = ttk.Style(self)
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#005a9e")   
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Success.TButton", foreground="white", background="#28a745", font=("Segoe UI", 10, "bold"))
        style.map("Success.TButton", background=[('active', '#218838')])
        style.configure("Status.TLabel", font=("Segoe UI", 9))                       

    def _load_and_populate(self):
        """Carrega dados e popula a UI, tratando erros."""
        try:
            self.manager.load_config()  
            self._update_radio_buttons()                        
            active_db = self.manager.get_active_db()  
            if active_db:  
                self.selected_db.set(active_db)  
                                             
                self.status_label.config(text=f"Atualmente ativo: {active_db}", foreground="blue")
            else:   
                 self.status_label.config(text="Nenhum banco ativo detectado no arquivo.", foreground="orange")   
            self.save_button.config(state="normal")                              

        except FileNotFoundError as e:
             self.show_error(str(e))
             self.save_button.config(state="disabled")  
        except Exception as e:   
             self.show_error(f"Erro inesperado ao carregar banco.ini: {e}")   
             self.save_button.config(state="disabled")

    def create_widgets(self, parent):                
                                
        ttk.Label(parent, text="Selecione a conexão de banco de dados para ATIVAR:", style="Header.TLabel").pack(anchor="w", pady=(0,5))

        self.radio_frame = ttk.LabelFrame(parent, text="Conexões Encontradas", relief="sunken", borderwidth=1, padding=15)   
        self.radio_frame.pack(fill="both", expand=True, pady=10)

        self.save_button = ttk.Button(parent, text="Salvar e Ativar Selecionado", command=self.save_selection, style="Success.TButton", state="disabled")                        
        self.save_button.pack(pady=5, fill="x", ipady=5)

        reload_button = ttk.Button(parent, text="Recarregar banco.ini", command=self._load_and_populate)
        reload_button.pack(pady=5, fill="x")

        self.status_label.pack(pady=(10, 0))  

    def _update_radio_buttons(self):   
         """Limpa e recria os radio buttons com base nos dados carregados."""
                                      
         for widget in self.radio_frame.winfo_children():
              widget.destroy()

         all_dbs = self.manager.get_all_dbs()  

         if not all_dbs:  
             ttk.Label(self.radio_frame, text="Nenhum bloco de configuração válido foi encontrado.", foreground="red").pack()   
             self.save_button.config(state="disabled")  
             return  

         for db_name in all_dbs:  
                                  
             rb = ttk.Radiobutton(  
                 self.radio_frame,  
                 text=db_name,   
                 variable=self.selected_db,   
                 value=db_name   
             )
             rb.pack(anchor="w", pady=2)

    def save_selection(self):  
        chosen_db = self.selected_db.get()   
        if not chosen_db:  
            self.show_error("Nenhum banco de dados foi selecionado.")  
            return  

        try:  
            self.manager.activate_db(chosen_db)  
            self.manager.save_config()  
            self.status_label.config(text=f"Sucesso! '{chosen_db}' agora está ativo no arquivo banco.ini.", foreground="green")

            self.manager.load_config()  
                                     
            self.selected_db.set(self.manager.get_active_db())  
                                                                              
            self._update_radio_buttons()   

        except Exception as e:  
            self.show_error(f"Erro ao salvar: {e}")  

    def show_error(self, message):  
        self.status_label.config(text=message, foreground="red")  
        messagebox.showerror("Erro", message, parent=self)

if __name__ == "__main__":  
                                                           
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    root_dir = os.path.dirname(script_dir)   
    ini_path = os.path.join(root_dir, "banco.ini")

    if not os.path.exists(ini_path):  
        try:  
                                                         
             with open(ini_path, 'w', encoding='utf-8') as f:
                f.write("""[database]
# ======================================================================
# INSTRUÇÕES DE USO:
# Para usar um banco de dados, descomente a seção correspondente
# (removendo o '#' do início de cada linha) e comente as outras.
# Apenas UMA configuração de banco de dados deve estar ativa por vez.
# ======================================================================

# --- SQLite (Exemplo Ativo) ---
type = sqlite
path = meu_banco.db

# --- PostgreSQL (Exemplo Comentado) ---
# type = postgresql
# host = localhost
# port = 5432
# dbname = meu_banco_pg
# user = meu_usuario
# password = minha_senha

# --- MySQL (Exemplo Comentado) ---
# type = mysql
# host = localhost
# port = 3306
# dbname = meu_banco_mysql
# user = meu_usuario
# password = minha_senha
""")
        except Exception as e:   
             messagebox.showerror("Erro Crítico", f"Não foi possível criar 'banco.ini' em {ini_path}: {e}")   
             exit()

    config_manager = DatabaseConfigManager(filepath=ini_path)  
    app = App(config_manager)  
    app.mainloop()  