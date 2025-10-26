# ✨ NexlifyStreamlit ✨

<!-- Badges de Status - Substitua as URLs -->

<div align="center">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/framework-Streamlit-red" alt="Framework">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/arquitetura-MVP%2520%2526%2520Camadas-brightgreen" alt="Architecture">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/licen%25C3%25A7a-MIT-lightgrey" alt="License">
</div>

<h1 align="center">✨ NexlifyStreamlit ✨</h1>

<p align="center">
<strong>Um framework web moderno sobre Streamlit, construído com uma arquitetura de software limpa, em camadas e orientada a padrões de projeto para máxima manutenibilidade, testabilidade e flexibilidade.</strong>
</p>

NexlifyStreamlit é a evolução de uma robusta aplicação desktop para uma plataforma web interativa. O projeto serve como um boilerplate avançado, demonstrando como aplicar princípios de engenharia de software (como MVP, Repository Pattern e Service Layer) em um ambiente reativo como o Streamlit, garantindo que a lógica de negócios e a camada de persistência permaneçam completamente agnósticas à tecnologia da interface.

Este documento é a referência canônica do projeto, servindo como um guia completo para usuários, desenvolvedores e arquitetos de sistema.

## 📜 Índice

- [🎯 Filosofia e Recursos Principais](#-filosofia-e-recursos-principais)
- [🏛️ Arquitetura Aprofundada](#️-arquitetura-aprofundada)
- [🛠️ Stack Tecnológico](#️-stack-tecnológico)
- [🚀 Guia de Instalação Rápida](#-guia-de-instalação-rápida)
- [🏃 Executando a Aplicação](#-executando-a-aplicação)
- [🔧 Como Estender: Adicionando Novas Páginas](#-como-estender-adicionando-novas-páginas)
- [⚙️ Modos de Operação e Ferramentas](#️-modos-de-operação-e-ferramentas)
- [🤝 Como Contribuir](#-como-contribuir)
- [📄 Licença](#-licença)

## 🎯 Filosofia e Recursos Principais

Arquitetura Limpa (MVP Adaptado): Separação estrita entre Interface (View), Lógica de UI (Controller/Presenter) e serviços de back-end, promovendo baixo acoplamento e alta coesão.

Agnóstico de Banco de Dados (Padrão Repository): Suporte nativo para SQLite, PostgreSQL, MariaDB/MySQL e SQL Server através do SQLAlchemy Core, isolando o núcleo da aplicação dos detalhes do SGBD.

Transações Atômicas (ACID): A Camada de Serviço (DataService) gerencia operações de negócio complexas de forma atômica, garantindo a integridade dos dados e a rastreabilidade através de logs de auditoria.

Segurança End-to-End:

Autenticação e RBAC (Role-Based Access Control): Sistema de login seguro com hashing bcrypt e controle de acesso por perfil de usuário.

Criptografia de Credenciais: Credenciais do banco de dados são criptografadas em repouso com Fernet (AES-128).

Configuração Dinâmica com Feature Flags: O config_settings.ini permite alternar entre modos de Produção, Desenvolvimento (Back-end) e Desenvolvimento (Front-end/Offline).

Ferramentas de Desenvolvedor: Um conjunto de ferramentas GUI (instalacao/) simplifica a configuração do ambiente, a geração de credenciais e a edição de flags.

Editor de Tema Dinâmico: Permite a personalização visual da UI em tempo de execução, com salvamento das configurações para persistência.

## 🏛️ Arquitetura Aprofundada

A arquitetura foi concebida para maximizar o desacoplamento. Cada camada tem uma responsabilidade única, permitindo que a UI ou o SGBD sejam trocados com impacto mínimo no núcleo de negócios.

```mermaid
graph TD
    subgraph "Apresentação (View)"
        A[Interface Streamlit <br> `pages/*.py` <br> `components/*_view.py`]
    end
    subgraph "Controle (Controller/Presenter)"
        B[Lógica de Estado da UI <br> `components/*_controller.py`]
    end
    subgraph "Serviço (Service)"
        C[Orquestração de Transações <br> `persistencia/data_service.py`]
    end
    subgraph "Repositório (Repository)"
        D[Abstração de Acesso a Dados (CRUD) <br> `persistencia/repository.py`]
    end
    subgraph "Infraestrutura"
        E[Engine, Segurança, Auth <br> `persistencia/database.py` <br> `persistencia/security.py`]
    end
    subgraph "Banco de Dados"
        F[(SGBD <br> SQLite / PostgreSQL / etc.)]
    end

    A -- Interação do Usuário --> B
    B -- Chama Lógica Complexa --> C
    B -- Chama CRUD Simples --> D
    C -- Usa Múltiplas Operações --> D
    D -- Executa Queries --> E
    E -- Conecta e Transaciona --> F
```

| Camada | Módulos Principais | Padrões de Projeto Aplicados | Responsabilidade Primária |
|--------|--------------------|------------------------------|---------------------------|
| Apresentação (UI) | pages/*.py, components/*_view.py | Model-View-Presenter (MVP) Adaptado, Front Controller | Renderização da UI Streamlit e captura de eventos do usuário. |
| Controle (Presenter) | components/*_controller.py | Presenter, Session State Manager | Gerenciamento de st.session_state e mapeamento de eventos da UI para a Camada de Serviço ou Repositório. |
| Serviço (Domínio) | persistencia/data_service.py | Service Layer, Transação Atômica | Orquestração da lógica de negócio complexa e garantia da integridade transacional (ACID). |
| Repositório | persistencia/repository.py | Repository, Data Mapper (via SQLAlchemy) | Interface CRUD genérica que abstrai completamente a fonte de dados, desacoplando o negócio do SQL. |
| Infraestrutura | persistencia/database.py, security.py, auth.py | Strategy, Adapter, Singleton (de-facto) | Gerenciamento de conexão (Multi-Banco), segurança (hashing/criptografia) e logging. |

## 🛠️ Stack Tecnológico

| Categoria | Tecnologia | Propósito e Justificativa |
|-----------|------------|---------------------------|
| Frontend | Streamlit | Framework web para rápido desenvolvimento de UIs interativas com Python. |
| Backend | Python 3.8+ | Linguagem principal, unificando todo o desenvolvimento. |
| Acesso a Dados | SQLAlchemy Core 2.0 | Acesso a dados agnóstico de SGBD, mantendo controle total sobre as queries SQL. |
| Estrutura de Dados | Pandas | Manipulação de DTOs (Data Transfer Objects) e integração nativa com Streamlit. |
| Segurança (Usuário) | bcrypt | Hashing de senhas, padrão da indústria por sua robustez contra força bruta. |
| Segurança (Config) | cryptography (Fernet) | Criptografia simétrica (AES-128) para proteger credenciais em repouso. |
| Dev Tools | Tkinter | Ferramentas gráficas de apoio para configuração do ambiente de desenvolvimento. |

## 🚀 Guia de Instalação Rápida

### Clone o Repositório:

```bash
git clone <URL_DO_REPOSITORIO> && cd <NOME_DA_PASTA>
```

### Crie e Ative um Ambiente Virtual:

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### Instale as Dependências:

Crie um arquivo `requirements.txt` com o conteúdo abaixo e execute `pip install -r requirements.txt`.

```
streamlit
pandas
sqlalchemy
bcrypt
cryptography
# Adicione o driver do seu banco de dados:
# psycopg2-binary
# PyMySQL
# mariadb
# pymssql
```

### Configure o Ambiente (Método Fácil):

Utilize as ferramentas gráficas fornecidas para uma configuração rápida e sem erros.

```bash
python instalacao/launch_devtools.py
```

- Configure a Conexão (banco.ini): Selecione o SGBD que deseja usar.
- Gere Credenciais/Hashes: Criptografe as senhas para o banco.ini e gere hashes para os usuários nos scripts .sql.
- Ajuste as Flags (config_settings.ini): Altere os modos de operação da aplicação.

### Prepare o Banco de Dados:

Localize o script .sql correspondente ao seu SGBD na pasta `persistencia/`.

Execute este script em seu SGBD para criar as tabelas e inserir os dados iniciais.

## 🏃 Executando a Aplicação

Com tudo configurado, inicie o servidor Streamlit:

```bash
streamlit run Home.py
```

A aplicação será aberta no seu navegador. As credenciais padrão são `admin / admin123`.

## 🔧 Como Estender: Adicionando Novas Páginas

Adicionar uma nova funcionalidade (página) é um processo padronizado que segue a arquitetura do projeto.

1. **Crie o Controller e a View:**

   Na pasta `components/`, crie seus arquivos `seu_painel_controller.py` e `seu_painel_view.py`.

   Dica: Copie e renomeie os arquivos `painel_modelo_controller.py` e `painel_modelo_view.py` como ponto de partida.

   Desenvolva a lógica de UI no Controller e a renderização dos widgets na View.

2. **Crie o Ponto de Entrada da Página:**

   Na pasta `pages/`, crie um novo arquivo Python. O nome do arquivo define o título e a ordem no menu lateral.

   Exemplo: `pages/9_⭐_Seu_Painel.py`.

3. **Adicione o Código Padrão na Página:**

   Cole o seguinte código no seu novo arquivo de página, adaptando os nomes e perfis de acesso:

   ```python
   # templates/9_⭐_Seu_Painel.py

   import streamlit as st
   from utils.st_utils import st_check_session, check_access
   from components.seu_painel_controller import SeuPainelController

   # 1. Configura a página
   st.set_page_config(page_title="Seu Painel", layout="wide")

   # 2. Garante que o usuário está logado
   st_check_session()

   # 3. Restringe o acesso a perfis específicos (lista vazia para acesso público)
   check_access(['Administrador Global', 'Gerente de TI'])

   # 4. Instancia e executa o controller
   controller = SeuPainelController()
   controller.run()
   ```

## ⚙️ Modos de Operação e Ferramentas

O `config_settings.ini` permite alternar o comportamento do sistema para diferentes cenários:

- **🌎 Produção:** `database_enabled=True, use_login=True`. Máxima segurança e funcionalidade.
- **🛠️ Desenvolvimento (Back-end):** `database_enabled=True, use_login=False`. Pula o login para testes rápidos de funcionalidades que dependem de dados.
- **🎨 Desenvolvimento (Front-end/Offline):** `database_enabled=False, use_login=False`. Permite trabalhar na UI sem depender de um banco de dados ativo.

O sistema possui uma validação de consistência em `main.py` que impede a inicialização com combinações de flags ilógicas.


## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

*Copie o conteúdo acima e salve como `README.md` em seu repositório para download ou uso local.*