"""
=============================================================================
INSTALADOR DE DEPENDÊNCIAS DE FRONTEND (Executar de /instalacao)
=============================================================================

Este script baixa as bibliotecas CSS e JavaScript necessárias
(Bootstrap, HTMX, ApexCharts) da internet e
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

SCRIPT_DIR = pathlib.Path(__file__).parent

PROJECT_ROOT = SCRIPT_DIR.parent

BASE_DIR = PROJECT_ROOT / "static"
CSS_DIR = BASE_DIR / "css"
JS_DIR = BASE_DIR / "js"

LIBS_TO_DOWNLOAD = [

    {
        "name": "Bootstrap CSS",
        "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
        "file": "bootstrap.min.css",
        "target_dir": CSS_DIR
    },
    {
        "name": "Bootstrap JS",
        "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
        "file": "bootstrap.bundle.min.js",
        "target_dir": JS_DIR
    },
    {
        "name": "HTMX",
        "url": "https://cdn.jsdelivr.net/npm/htmx.org@2.0.0/dist/htmx.min.js",
        "file": "htmx.min.js",
        "target_dir": JS_DIR
    },
    # Alpine.js removido
    # Plotly removido
    {
        "name": "ApexCharts",
        "url": "https://cdn.jsdelivr.net/npm/apexcharts",
        "file": "apexcharts.min.js",
        "target_dir": JS_DIR
    }
    # Hyperscript removido
]

def download_libs():
    """
    Baixa e salva as bibliotecas de frontend nas pastas corretas na raiz do projeto.
    """
    print(f"--- Iniciando Verificação de Bibliotecas de Frontend ---")
    print(f"Executando de: '{SCRIPT_DIR.resolve()}'")
    print(f"Salvando em: '{BASE_DIR.resolve()}'")

    dirs_to_create = {CSS_DIR, JS_DIR}
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Diretório de destino verificado: '{dir_path.resolve()}'")
        except Exception as e:
            print(f"ERRO: Não foi possível criar o diretório '{dir_path}': {e}", file=sys.stderr)
            return

    total_downloaded = 0
    for lib in LIBS_TO_DOWNLOAD:
        target_path = lib["target_dir"] / lib["file"]

        if target_path.exists():
            print(f"[OK] '{lib['name']}' ({target_path}) já existe. Pulando.")
            continue

        print(f"[DOWNLOAD] Baixando '{lib['name']}' de '{lib['url']}'...")
        try:
            response = requests.get(lib['url'], allow_redirects=True, timeout=10)
                                                     
            response.raise_for_status()

            target_path.write_bytes(response.content)
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