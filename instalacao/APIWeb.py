"""
=============================================================================
INSTALADOR DE DEPENDÊNCIAS DE FRONTEND (Executar de /instalacao)
=============================================================================

Este script baixa TODAS as bibliotecas CSS e JavaScript necessárias
(Bootstrap, HTMX, Alpine, Plotly, ApexCharts, Hyperscript) da internet e
as salva nas pastas 'static/css/' e 'static/js/' LOCALIZADAS NA RAIZ DO PROJETO.

Este script DEVE ser executado de dentro da pasta 'instalacao'.

Execute este script uma vez para configurar o ambiente:
   cd instalacao
   python APIWeb.py

Este script requer a biblioteca 'requests'. Se você não a tiver:
   pip install requests
=============================================================================
"""

import requests
import pathlib
import sys
# Não precisamos mais de 'os' se usarmos __file__

# --- CORREÇÃO: Define caminhos relativos à localização do script ---
# Encontra o diretório onde este script (APIWeb.py) está localizado (instalacao)
SCRIPT_DIR = pathlib.Path(__file__).parent
# Define a pasta RAIZ do projeto (o diretório PAI da pasta 'instalacao')
PROJECT_ROOT = SCRIPT_DIR.parent
# Define a pasta 'static' dentro da raiz do projeto
BASE_DIR = PROJECT_ROOT / "static"
CSS_DIR = BASE_DIR / "css"
JS_DIR = BASE_DIR / "js"
# --- FIM DA CORREÇÃO ---


# Lista de bibliotecas para baixar
LIBS_TO_DOWNLOAD = [
    # -- Bibliotecas Base (Bootstrap, HTMX, Alpine) --
    {
        "name": "Bootstrap CSS",
        "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
        "file": "bootstrap.min.css",
        "target_dir": CSS_DIR # Usa o caminho absoluto calculado
    },
    {
        "name": "Bootstrap JS",
        "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
        "file": "bootstrap.bundle.min.js",
        "target_dir": JS_DIR # Usa o caminho absoluto calculado
    },
    {
        "name": "HTMX",
        "url": "https://cdn.jsdelivr.net/npm/htmx.org@2.0.0/dist/htmx.min.js",
        "file": "htmx.min.js",
        "target_dir": JS_DIR # Usa o caminho absoluto calculado
    },
    {
        "name": "Alpine.js",
        "url": "https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js",
        "file": "alpine.min.js",
        "target_dir": JS_DIR # Usa o caminho absoluto calculado
    },

    # -- Bibliotecas de Gráficos e Extras --
    {
        "name": "Plotly",
        "url": "https://cdn.plot.ly/plotly-latest.min.js",
        "file": "plotly-latest.min.js",
        "target_dir": JS_DIR # Usa o caminho absoluto calculado
    },
    {
        "name": "ApexCharts",
        "url": "https://cdn.jsdelivr.net/npm/apexcharts",
        "file": "apexcharts.min.js",
        "target_dir": JS_DIR # Usa o caminho absoluto calculado
    },
    {
        "name": "Hyperscript",
        "url": "https://unpkg.com/hyperscript.org@0.9.12/dist/_hyperscript.min.js",
        "file": "_hyperscript.min.js",
        "target_dir": JS_DIR
    }
]

def download_libs():
    """
    Baixa e salva as bibliotecas de frontend nas pastas corretas na raiz do projeto.
    """
    print(f"--- Iniciando Verificação de Bibliotecas de Frontend ---")
    print(f"Executando de: '{SCRIPT_DIR.resolve()}'")
    print(f"Salvando em: '{BASE_DIR.resolve()}'")

    # 1. Garante que os diretórios existam
    dirs_to_create = {CSS_DIR, JS_DIR}
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Diretório de destino verificado: '{dir_path.resolve()}'")
        except Exception as e:
            print(f"ERRO: Não foi possível criar o diretório '{dir_path}': {e}", file=sys.stderr)
            return

    # 2. Baixa cada biblioteca
    total_downloaded = 0
    for lib in LIBS_TO_DOWNLOAD:
        target_path = lib["target_dir"] / lib["file"]

        # Verifica se o arquivo já existe
        if target_path.exists():
            print(f"[OK] '{lib['name']}' ({target_path}) já existe. Pulando.")
            continue

        # Se não existe, tenta baixar
        print(f"[DOWNLOAD] Baixando '{lib['name']}' de '{lib['url']}'...")
        try:
            response = requests.get(lib['url'], allow_redirects=True, timeout=10)
            # Verifica se o download foi bem-sucedido
            response.raise_for_status()

            # Salva o conteúdo no arquivo
            target_path.write_text(response.text, encoding='utf-8')
            print(f"   -> Salvo com sucesso em '{target_path}'")
            total_downloaded += 1

        except requests.exceptions.RequestException as e:
            print(f"   ERRO: Falha ao baixar '{lib['name']}': {e}", file=sys.stderr)
        except Exception as e:
            print(f"   ERRO: Falha ao salvar '{lib['name']}': {e}", file=sys.stderr)

    print("--- Verificação Concluída ---")
    if total_downloaded > 0:
        print(f"{total_downloaded} nova(s) biblioteca(s) baixada(s).")
    else:
        print("Todas as bibliotecas locais já estavam atualizadas.")

if __name__ == "__main__":
    download_libs()