import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox  
from pathlib import Path
import sys

try:
                                                                                                 
    INSTALL_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = INSTALL_DIR.parent.resolve()                                       
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))                                     

    from persistencia.security import load_key, encrypt_message
    from persistencia.auth import hash_password
except ImportError as e:
    messagebox.showerror("Erro Crítico de Importação",  
                         f"Não foi possível importar os módulos de segurança: {e}\n\n"  
                         f"Certifique-se de que este script está na pasta 'instalacao' "  
                         f"e que a pasta 'persistencia' existe na raiz do projeto.")  
    sys.exit(1)

APP_TITLE = "Gerador de Credenciais"                           
FONT_DEFAULT = ("Segoe UI", 10)     
FONT_BOLD = ("Segoe UI", 10, "bold")     
FONT_CODE = ("Courier New", 9)                                          
FONT_H1 = ("Segoe UI", 16, "bold")     
FONT_H2 = ("Segoe UI", 12, "bold")     

class CredentialToolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)  
        self.geometry("800x700")  
        self.minsize(700, 600)  

        self._setup_styles()   

        notebook = ttk.Notebook(self)  
        notebook.pack(fill="both", expand=True, padx=10, pady=10)  

        ini_tab = ttk.Frame(notebook, padding=15)   
        sql_tab = ttk.Frame(notebook, padding=15)   
        tutorial_tab = ttk.Frame(notebook, padding=15)   

        notebook.add(ini_tab, text=" 🔑 Gerador .ini ")  
        notebook.add(sql_tab, text=" #️⃣ Gerador Hash .sql ")  
        notebook.add(tutorial_tab, text=" ❓ Como Usar ")   

        self._create_ini_tab(ini_tab)  
        self._create_sql_tab(sql_tab)   
        self._create_tutorial_tab(tutorial_tab)   

    def _setup_styles(self):  
        self.style = ttk.Style(self)  
        self.style.theme_use('clam')  
        self.style.configure('TLabel', font=FONT_DEFAULT)  
        self.style.configure('TButton', font=FONT_DEFAULT, padding=5)                              
        self.style.configure('TNotebook.Tab', font=FONT_BOLD, padding=[10, 5])                       
        self.style.configure('TLabelframe.Label', font=FONT_H2, foreground="#005a9e")                    
        self.style.configure('Success.TButton', foreground="white", background="#28a745", font=FONT_BOLD)                 
        self.style.map("Success.TButton", background=[('active', '#218838')])

    def _create_ini_tab(self, parent):  
                                
        main_ini_frame = ttk.Frame(parent)  
        main_ini_frame.pack(fill='both', expand=True)

        input_frame = ttk.LabelFrame(main_ini_frame, text="Credenciais para banco.ini", padding=15)   
        input_frame.pack(fill="x", pady=(0, 10))  
        input_frame.columnconfigure(1, weight=1)  

        ttk.Label(input_frame, text="Usuário:").grid(row=0, column=0, padx=5, pady=5, sticky="w")  
        self.ini_user_entry = ttk.Entry(input_frame, width=40, font=FONT_DEFAULT)  
        self.ini_user_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")  

        ttk.Label(input_frame, text="Senha:").grid(row=1, column=0, padx=5, pady=5, sticky="w")  
        self.ini_pass_entry = ttk.Entry(input_frame, width=40, show="*", font=FONT_DEFAULT)  
        self.ini_pass_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")   

        self.ini_show_pass_var = tk.BooleanVar()  
        ini_show_pass_check = ttk.Checkbutton(input_frame, text="Mostrar senha", variable=self.ini_show_pass_var,  
                                              command=self._toggle_ini_pass)  
        ini_show_pass_check.grid(row=2, column=1, padx=5, pady=(0, 10), sticky="w")   

        ttk.Button(input_frame, text="Gerar Credenciais", command=self._generate_ini_creds, style="Success.TButton").grid(row=3, column=0,                          
                                                                                           columnspan=2, pady=10,   
                                                                                           sticky="ew", ipady=5)

        output_frame = ttk.LabelFrame(main_ini_frame, text="Resultado (Copie e cole no banco.ini)", padding=15)  
        output_frame.pack(fill="both", expand=True, pady=(10, 0))  
        output_frame.rowconfigure(0, weight=1)  
        output_frame.columnconfigure(0, weight=1)  

        self.ini_output_area = scrolledtext.ScrolledText(output_frame, height=5, font=FONT_CODE, wrap=tk.WORD, relief="solid", borderwidth=1)                         
        self.ini_output_area.grid(row=0, column=0, sticky="nsew")  
        self.ini_output_area.config(state="disabled")  

        ttk.Button(output_frame, text="Copiar Resultado", command=self._copy_ini_output).grid(row=1, column=0,     
                                                                                              pady=(10, 0), sticky="ew", ipady=5)                               

    def _generate_ini_creds(self):  
        user = self.ini_user_entry.get()   
        password = self.ini_pass_entry.get()   

        if not user or not password:   
            messagebox.showwarning("Campos Vazios", "Por favor, preencha o usuário e a senha.", parent=self)  
            return  

        try:  
            key = load_key()  
            encrypted_user = encrypt_message(user, key)   
            encrypted_password = encrypt_message(password, key)   
            output_text = f"user = {encrypted_user}\npassword = {encrypted_password}"   

            self.ini_output_area.config(state="normal")  
            self.ini_output_area.delete("1.0", tk.END)  
            self.ini_output_area.insert("1.0", output_text)  
            self.ini_output_area.config(state="disabled")   

        except Exception as e:  
            messagebox.showerror("Erro de Criptografia", f"Não foi possível gerar as credenciais:\n{e}", parent=self)   

    def _copy_ini_output(self):  
        self._copy_to_clipboard(self.ini_output_area.get("1.0", tk.END))  

    def _toggle_ini_pass(self):  
        show = "" if self.ini_show_pass_var.get() else "*"  
        self.ini_pass_entry.config(show=show)

    def _create_sql_tab(self, parent):   
                                
        main_sql_frame = ttk.Frame(parent)   
        main_sql_frame.pack(fill='both', expand=True)

        input_frame = ttk.LabelFrame(main_sql_frame, text="Hash de Senha para .sql (Bcrypt)", padding=15)  
        input_frame.pack(fill="x", pady=(0, 10))  
        input_frame.columnconfigure(1, weight=1)  

        ttk.Label(input_frame, text="Senha:").grid(row=0, column=0, padx=5, pady=5, sticky="w")  
        self.sql_pass_entry = ttk.Entry(input_frame, width=40, show="*", font=FONT_DEFAULT)  
        self.sql_pass_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")   

        self.sql_show_pass_var = tk.BooleanVar()  
        sql_show_pass_check = ttk.Checkbutton(input_frame, text="Mostrar senha", variable=self.sql_show_pass_var,  
                                             command=self._toggle_sql_pass)  
        sql_show_pass_check.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="w")                      

        ttk.Button(input_frame, text="Gerar Hash", command=self._generate_sql_hash, style="Success.TButton").grid(row=2, column=0, columnspan=2,                         
                                                                                       pady=10, sticky="ew", ipady=5)

        output_frame = ttk.LabelFrame(main_sql_frame, text="Resultado (Copie e cole no .sql)", padding=15)  
        output_frame.pack(fill="x", pady=(10, 0))  
        output_frame.columnconfigure(0, weight=1)  

        self.sql_output_entry = ttk.Entry(output_frame, font=FONT_CODE, state="readonly")                               
        self.sql_output_entry.grid(row=0, column=0, sticky="ew", pady=(0, 5))                       

        ttk.Button(output_frame, text="Copiar Hash", command=self._copy_hash_output).grid(row=1, column=0, pady=(5, 0),     
                                                                                          sticky="ew", ipady=5)                        

    def _generate_sql_hash(self):  
        password = self.sql_pass_entry.get()  
        if not password:  
            messagebox.showwarning("Campo Vazio", "Por favor, digite uma senha para gerar o hash.", parent=self)   
            return   

        try:  
            generated_hash = hash_password(password)  
            self.sql_output_entry.config(state="normal")  
            self.sql_output_entry.delete(0, tk.END)  
            self.sql_output_entry.insert(0, generated_hash)   
            self.sql_output_entry.config(state="readonly")   
        except Exception as e:  
            messagebox.showerror("Erro ao Gerar Hash", f"Não foi possível gerar o hash:\n{e}", parent=self)   

    def _copy_hash_output(self):  
        self._copy_to_clipboard(self.sql_output_entry.get())  

    def _toggle_sql_pass(self):  
        show = "" if self.sql_show_pass_var.get() else "*"  
        self.sql_pass_entry.config(show=show)

    def _create_tutorial_tab(self, parent):   
        parent.rowconfigure(0, weight=1)  
        parent.columnconfigure(0, weight=1)  

        text_area = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=FONT_DEFAULT, relief="flat", padx=10, bd=0)                    
        text_area.grid(row=0, column=0, sticky="nsew")

        tutorial_text = """Bem-vindo ao Gerador de Credenciais! # 
Este programa gera DOIS tipos diferentes de senhas. É essencial entender a diferença. # 

O Conceito Mais Importante: Chave vs. Selo # 
Imagine que sua aplicação tem duas necessidades de segurança: #

1. A Chave da Casa (Aba 1): #
A aplicação (Streamlit) precisa de uma "chave" para "abrir a porta" do banco de dados (PostgreSQL, MariaDB, etc.). # 
• Tipo: Criptografia (Reversível). # 
• Como funciona: Usamos uma "chave mestra" (o arquivo secret.key) para trancar (criptografar) a senha no arquivo banco.ini. # 
A aplicação usa a mesma chave mestra para destrancar (descriptografar) e poder usar a senha. # 

2. O Selo de Identidade (Aba 2): # 
Um usuário (ex: 'admin') precisa provar quem é para entrar na *aplicação*. # 
• Tipo: Hashing (Irreversível). # 
• Como funciona: O usuário digita a senha (ex: 'admin123'). # 
O sistema NÃO armazena 'admin123'. Ele a transforma num "selo" (hash) único, que não pode ser revertido. # 
Quando o usuário tenta logar, o sistema cria um novo selo da senha digitada e compara com o selo guardado no banco. # 
Se os selos baterem, o login é permitido. # 

-------------------------------------------------- #

Guia da Aba 1: Gerador .ini (A Chave da Casa) #
Use isto para a senha que a APLICAÇÃO usa para se conectar ao BANCO DE DADOS. # 

Quando Usar: #
Para configurar o arquivo banco.ini quando você usa um banco de dados como PostgreSQL, MariaDB, SQL Server, etc. (Não é usado para SQLite). # 

Passo a Passo: #
1.  Na 'Aba 1', digite o usuário e a senha REAIS do seu banco de dados (ex: o usuário 'gato' e a senha '-Vladmir!5Anos-' do seu MariaDB). # 
2.  Clique em "Gerar Credenciais". #
3.  Isto irá (re)usar ou criar um arquivo chamado secret.key na pasta do projeto. # 
4.  Copie o resultado (as duas linhas que começam com user = ... e password = ...). # 
5.  Abra seu arquivo banco.ini. #
6.  Cole as linhas copiadas na seção do banco de dados que você ativou (ex: [mariadb]). # 

Exemplo de banco.ini: #
[mariadb] #
type = mariadb #
host = 127.0.0.1 #
dbname = nexlifyttk # (Exemplo com nome minúsculo)
; Cole o resultado da Aba 1 aqui # 
user = gAAAAABnQ... #
password = gAAAAABnQ... #

ALERTA IMPORTANTE: #
Nunca delete o arquivo secret.key! # 
Se você o perder, a aplicação não conseguirá mais ler a senha no banco.ini e tudo vai parar de funcionar. # 
Guarde-o com segurança. # 

-------------------------------------------------- #

Guia da Aba 2: Gerador .sql (O Selo de Identidade) #
Use isto para as senhas que os USUÁRIOS (admin, diretor, etc.) usam para LOGAR NA APLICAÇÃO. # 

Quando Usar: #
Para preencher a coluna 'senha_criptografada' na sua tabela 'usuarios', geralmente no arquivo sql_schema_sqllite.sql. #  (Corrigido para minúsculas)

Passo a Passo: #
1.  Decida qual será a senha de um usuário. (Ex: O usuário 'admin' terá a senha 'senhaForte123'). # 
2.  Na 'Aba 2', digite esta senha ('senhaForte123'). # 
3.  Clique em "Gerar Hash". #
4.  Copie o resultado. Será um texto longo que começa com $2b$... (isto é um hash Bcrypt). # 
5.  Abra seu arquivo .sql (ex: sql_schema_sqllite.sql). # 
6.  Encontre o comando INSERT para o usuário 'admin'. # 
7.  Cole o hash gerado no lugar da senha antiga, dentro das aspas. # 

Exemplo de sql_schema_sqllite.sql: # (Corrigido para minúsculas)
INSERT INTO usuarios (login_usuario, senha_criptografada, nome_completo, tipo_acesso)  # 
VALUES # 
('admin', 'COLE_O_HASH_GERADO_AQUI', 'Usuário Administrador', 'Administrador Global'); # 

Este hash é seguro. Mesmo que alguém o veja, não é possível descobrir a senha original 'senhaForte123' a partir dele. # 
"""  
        text_area.insert("1.0", tutorial_text)

        text_area.tag_configure("h1", font=FONT_H1, spacing3=10, spacing1=10)  
        text_area.tag_configure("h2", font=FONT_H2, foreground="#005A9E", spacing3=5, spacing1=10)  
        text_area.tag_configure("bold", font=FONT_BOLD)  
        text_area.tag_configure("alert", font=FONT_BOLD, foreground="#800020")  
        text_area.tag_configure("code", font=FONT_CODE, background="#f0f0f0", relief="solid", borderwidth=1)  
        text_area.tag_configure("line", overstrike=True)

        text_area.tag_configure("sql_code", font=FONT_CODE)

        start_index = "1.0"
        while True:
            pos = text_area.search("'usuarios'", start_index, stopindex=tk.END, regexp=True)
            if not pos: break
            end_index = f"{pos}+{len("'usuarios'")}c"
            text_area.tag_add("sql_code", pos, end_index)
            start_index = end_index

        start_index = "1.0"
        while True:
            pos = text_area.search("'senha_criptografada'", start_index, stopindex=tk.END, regexp=True)
            if not pos: break
            end_index = f"{pos}+{len("'senha_criptografada'")}c"
            text_area.tag_add("sql_code", pos, end_index)
            start_index = end_index

        start_index = "1.0"
        while True:
            pos = text_area.search("sql_schema_sqllite.sql", start_index, stopindex=tk.END)
            if not pos: break
            end_index = f"{pos}+{len("sql_schema_sqllite.sql")}c"
            text_area.tag_add("code", pos, end_index)                              
            start_index = end_index

        text_area.config(state="disabled")

    def _copy_to_clipboard(self, content):  
        content = content.strip()  
        if content:  
            self.clipboard_clear()  
            self.clipboard_append(content)   
            messagebox.showinfo("Sucesso", "Conteúdo copiado para a área de transferências.", parent=self)  
        else:  
            messagebox.showwarning("Nada a Copiar", "Nenhum conteúdo foi gerado para copiar.", parent=self)

if __name__ == "__main__":  
    app = CredentialToolApp()  
    app.mainloop()  