# routers/iris.py
# (Importações e definições iniciais permanecem as mesmas)
import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from functools import lru_cache
import json # Mantido para _safe_json_dumps, embora possamos não precisar dele para ApexCharts

from fastapi import APIRouter, Request, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import auth as app_auth
import config
from utils.settings import get_current_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/iris",
    tags=["Análise Iris"],
    dependencies=[Depends(app_auth.get_current_user)]
)
templates = Jinja2Templates(directory="templates")

CSV_DIR = Path("csv")
FILE_PATH = CSV_DIR / 'iris_dataset.csv'

# --- INÍCIO DAS ALTERAÇÕES ---

EXPECTED_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target', 'species']
ORIGINAL_COLUMNS_MAP = {
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width'
}

@lru_cache(maxsize=1)
def load_and_prepare_data() -> pd.DataFrame:
    """
    Carrega o dataset Iris. Se não existir, baixa.
    SEMPRE garante que as colunas estejam no formato padronizado e salva em CSV.
    """
    df = None
    created_now = False

    if not FILE_PATH.exists():
        logger.warning(f"Arquivo {FILE_PATH} não encontrado, tentando criar...")
        if not CSV_DIR.exists():
            try:
                CSV_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Diretório '{CSV_DIR}' criado.")
            except OSError as e:
                logger.error(f"Falha crítica: Não foi possível criar o diretório '{CSV_DIR}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Erro ao criar diretório '{CSV_DIR}'.")
        try:
            iris = load_iris(as_frame=True)
            df = pd.concat([iris.data, iris.target], axis=1)
            # Renomeia imediatamente após carregar do sklearn
            df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
            created_now = True
            logger.info("Dataset Iris carregado do Scikit-learn.")
        except Exception as e:
            logger.error(f"Falha crítica: Não foi possível carregar o dataset Iris do Scikit-learn: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Erro ao carregar o dataset Iris.")
    else:
        try:
            df = pd.read_csv(FILE_PATH)
            logger.info(f"Dataset Iris carregado do CSV existente: {FILE_PATH}")
            # Verifica se as colunas JÁ TÊM o formato antigo e renomeia se necessário
            renamed = False
            cols_to_rename = {k: v for k, v in ORIGINAL_COLUMNS_MAP.items() if k in df.columns}
            if cols_to_rename:
                df.rename(columns=cols_to_rename, inplace=True)
                logger.warning(f"Colunas renomeadas do formato antigo para o novo: {cols_to_rename}")
                renamed = True

            # Garante que as colunas esperadas (sem as originais) estejam presentes
            missing_expected = [col for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'] if col not in df.columns]
            if missing_expected:
                 logger.error(f"CSV existente ({FILE_PATH}) não contém colunas esperadas: {missing_expected}. Exclua o arquivo e reinicie para recriá-lo.")
                 raise HTTPException(status_code=500, detail=f"CSV corrompido ou formato inesperado: {FILE_PATH.name}")

            # Se renomeou, salva o CSV corrigido
            if renamed:
                try:
                    df.to_csv(FILE_PATH, index=False)
                    logger.info(f"CSV atualizado em {FILE_PATH} com colunas padronizadas.")
                except Exception as e:
                    logger.error(f"Não foi possível salvar o CSV corrigido em {FILE_PATH}: {e}")
                    # Continua com o df em memória mesmo assim

        except Exception as e:
            logger.error(f"Erro ao ler ou processar o CSV {FILE_PATH}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo {FILE_PATH.name}.")

    # Garante que a coluna 'species' exista
    if 'species' not in df.columns and 'target' in df.columns:
        target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        df['species'] = df['target'].map(target_names).fillna('unknown')
        logger.info("Coluna 'species' adicionada ao DataFrame.") # [cite: 1385, 1401, 1434, 1439] Mesmo log, mas agora acontece sempre que necessário.
    elif 'species' not in df.columns:
         logger.error("Coluna 'target' não encontrada para criar 'species'.")
         raise HTTPException(status_code=500, detail="Dataset Iris inválido: falta coluna 'target'.")

    # Garante que todas as colunas esperadas existam antes de salvar (se criou agora) ou retornar
    final_missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if final_missing:
         logger.error(f"Após o processamento, colunas esperadas ainda estão faltando: {final_missing}")
         raise HTTPException(status_code=500, detail="Erro interno ao preparar colunas do dataset.")

    # Salva no CSV apenas se foi criado agora (já renomeado)
    if created_now:
        try:
            df.to_csv(FILE_PATH, index=False)
            logger.info(f"Dataset Iris salvo em {FILE_PATH} com colunas padronizadas.") # [cite: 1453] Mesmo log
        except Exception as e:
            logger.error(f"Não foi possível salvar o novo CSV em {FILE_PATH}: {e}")
            # Continua com o df em memória

    logger.info(f"Dataset Iris pronto ({len(df)} linhas). Colunas: {list(df.columns)}") # [cite: 1453, 1460] Mesmo log
    return df

# Função _safe_json_dumps pode ser mantida por segurança, mas não será usada diretamente nas chamadas abaixo
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if pd.isna(obj): return None
        return super(NpEncoder, self).default(obj)

def _safe_json_dumps(data):
    return json.dumps(data, cls=NpEncoder)


# Funções _get_apex_*: Modificar para retornar o DICIONÁRIO Python, não uma string JSON
def _get_apex_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    try:
        series = []
        # Garante que as colunas existem antes de tentar usar
        if x_col not in df.columns or y_col not in df.columns or 'species' not in df.columns:
             logger.error(f"Colunas ausentes para scatter plot '{title}': X='{x_col}', Y='{y_col}', Group='species'. Colunas disponíveis: {list(df.columns)}")
             return {'error': f'Colunas necessárias ausentes'}

        for species in df['species'].unique():
            species_df = df[df['species'] == species]
            # Converte para tipos nativos Python aqui para evitar problemas de serialização depois
            data = [[float(round(x, 2)), float(round(y, 2))] for x, y in zip(species_df[x_col], species_df[y_col])]
            series.append({'name': str(species), 'data': data}) # Garante que name é string

        chart_data = {
            'series': series,
            'chart': {'type': 'scatter', 'height': 350, 'zoom': {'enabled': True, 'type': 'xy'}},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'title': {'text': x_col}, 'tickAmount': 10},
            'yaxis': {'title': {'text': y_col}},
            'legend': {'position': 'top'}
        }
        # return json.loads(_safe_json_dumps(chart_data)) # REMOVIDO json.loads
        return chart_data # RETORNA dict
    except Exception as e:
        logger.error(f"Erro ao criar scatter plot {title}: {e}", exc_info=True) # Adicionado exc_info
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_bar(df_agg: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    try:
        if x_col not in df_agg.columns or y_col not in df_agg.columns:
             logger.error(f"Colunas ausentes para bar chart '{title}': X='{x_col}', Y='{y_col}'. Colunas disponíveis: {list(df_agg.columns)}")
             return {'error': f'Colunas necessárias ausentes'}

        chart_data = {
            'series': [{'name': y_col, 'data': [float(v) for v in df_agg[y_col].tolist()]}], # Converte para float
            'chart': {'type': 'bar', 'height': 350},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'categories': [str(c) for c in df_agg[x_col].tolist()]}, # Converte para string
            'plotOptions': {'bar': {'distributed': True}},
            'legend': {'show': False}
        }
        # return json.loads(_safe_json_dumps(chart_data)) # REMOVIDO json.loads
        return chart_data # RETORNA dict
    except Exception as e:
        logger.error(f"Erro ao criar bar chart {title}: {e}", exc_info=True) # Adicionado exc_info
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    try:
        if x_col not in df.columns or y_col not in df.columns:
             logger.error(f"Colunas ausentes para boxplot '{title}': X='{x_col}', Y='{y_col}'. Colunas disponíveis: {list(df.columns)}")
             return {'error': f'Colunas necessárias ausentes'}

        series_data = []
        for group_name in sorted(df[x_col].unique()):
            values = df[df[x_col] == group_name][y_col].dropna()
            if len(values) > 0:
                stats = values.describe()
                # Converte para tipos nativos Python
                series_data.append({
                    'x': str(group_name), # Garante string
                    'y': [
                        float(round(stats.get('min', 0), 2)),
                        float(round(stats.get('25%', 0), 2)),
                        float(round(stats.get('50%', 0), 2)),
                        float(round(stats.get('75%', 0), 2)),
                        float(round(stats.get('max', 0), 2))
                    ]
                })

        chart_data = {
            'series': [{'name': y_col, 'type': 'boxPlot', 'data': series_data}],
            'chart': {'type': 'boxPlot', 'height': 350},
            'title': {'text': title, 'align': 'left'}
        }
        # return json.loads(_safe_json_dumps(chart_data)) # REMOVIDO json.loads
        return chart_data # RETORNA dict
    except Exception as e:
        logger.error(f"Erro ao criar boxplot {title}: {e}", exc_info=True) # Adicionado exc_info
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_histogram(df: pd.DataFrame, x_col: str, title: str) -> dict:
    try:
        if x_col not in df.columns:
             logger.error(f"Coluna ausente para histograma '{title}': X='{x_col}'. Colunas disponíveis: {list(df.columns)}")
             return {'error': f'Coluna necessária ausente'}

        values = df[x_col].dropna()
        if len(values) == 0:
            return {'error': 'Sem dados para histograma'}

        counts, bins = np.histogram(values, bins=10)
        bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

        chart_data = {
            'series': [{'name': 'Frequência', 'data': [int(c) for c in counts.tolist()]}], # Converte para int
            'chart': {'type': 'bar', 'height': 350},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'categories': bin_labels, 'title': {'text': x_col}},
            'yaxis': {'title': {'text': 'Frequência'}},
            'legend': {'show': False}
        }
        # return json.loads(_safe_json_dumps(chart_data)) # REMOVIDO json.loads
        return chart_data # RETORNA dict
    except Exception as e:
        logger.error(f"Erro ao criar histograma {title}: {e}", exc_info=True) # Adicionado exc_info
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_pca_2d(df: pd.DataFrame, title: str) -> dict: # Renomeado para _pca_2d
    """Formata dados do PCA 2D (PC1 vs PC2) para ApexCharts."""
    try:
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        # Verifica se todas as colunas numéricas existem
        missing_numeric = [col for col in numeric_cols if col not in df.columns]
        if missing_numeric:
             logger.error(f"Colunas numéricas ausentes para PCA: {missing_numeric}. Colunas disponíveis: {list(df.columns)}")
             return {'error': f'Colunas para PCA ausentes'}
        if 'species' not in df.columns:
             logger.error(f"Coluna 'species' ausente para PCA. Colunas disponíveis: {list(df.columns)}")
             return {'error': f"Coluna 'species' ausente"}


        X = df[numeric_cols].dropna()
        y_species = df.loc[X.index, 'species']

        if len(X) < 2:
            logger.warning("Dados insuficientes para PCA 2D")
            return {'error': 'Dados insuficientes para PCA'}

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        var_explained = f"{title} (Var. PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})"

        pca_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
        pca_df['species'] = y_species.reset_index(drop=True)

        series = []
        for species in pca_df['species'].unique():
            species_df = pca_df[pca_df['species'] == species]
            # Converte para tipos nativos Python
            data = [[float(round(x, 3)), float(round(y, 3))] for x, y in zip(species_df['PC1'], species_df['PC2'])]
            series.append({'name': str(species), 'data': data}) # Garante string

        chart_data = {
            'series': series,
            'chart': {'type': 'scatter', 'height': 450, 'zoom': {'enabled': True, 'type': 'xy'}},
            'title': {'text': var_explained, 'align': 'left'},
            'xaxis': {'title': {'text': 'PC1'}},
            'yaxis': {'title': {'text': 'PC2'}},
            'legend': {'position': 'top'},
            'markers': {'size': 5}
        }
        logger.info(f"PCA 2D gerado com sucesso: {len(X)} amostras, variância: {pca.explained_variance_ratio_}") # [cite: 1481] Mesmo log
        # return json.loads(_safe_json_dumps(chart_data)) # REMOVIDO json.loads
        return chart_data # RETORNA dict
    except Exception as e:
        logger.error(f"Erro ao criar PCA 2D: {e}", exc_info=True) # Adicionado exc_info
        return {'error': f'Erro ao gerar PCA: {str(e)}'}

# --- FIM DAS ALTERAÇÕES NAS FUNÇÕES AUXILIARES ---

# Rota principal (sem alterações)
@router.get("/", response_class=HTMLResponse)
async def get_iris_page(request: Request):
    settings = get_current_settings()
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("iris.html", context)

# Rota da tabela (sem alterações lógicas, mas agora depende da load_and_prepare_data corrigida)
@router.get("/tabela", response_class=HTMLResponse)
async def get_tabela_iris(request: Request, page: int = Query(1, ge=1)):
    try:
        df = load_and_prepare_data() # Agora garante colunas corretas
        PAGE_SIZE = 15
        total_rows = len(df)
        total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
        current_page = max(1, min(page, total_pages))
        start_idx = (current_page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)
        df_page = df.iloc[start_idx:end_idx]

        headers = [str(col) for col in df.columns.tolist()]
        data = df_page.to_dict('records')

        context = {
            "request": request, "headers": headers, "data": data,
            "page": current_page, "total_pages": total_pages,
            "start_idx": start_idx + 1 if total_rows > 0 else 0,
            "end_idx": end_idx, "total_rows": total_rows,
        }
        # CORREÇÃO: Certifique-se que o nome do template está correto (incluindo 'partials/')
        return templates.TemplateResponse("partials/_iris_tabela.html", context) # [cite: 665] Usa o nome correto do arquivo
    except HTTPException as h_exc:
        raise h_exc # Re-raise HTTPExceptions (como as de load_and_prepare_data)
    except Exception as e:
        logger.error(f"Erro ao carregar tabela: {e}", exc_info=True)
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar dados: {str(e)}</div>", status_code=500)

# Rota dos gráficos EDA
@router.get("/eda_charts", response_class=HTMLResponse)
async def get_eda_charts(request: Request):
    try:
        df = load_and_prepare_data() # Garante colunas corretas
        logger.info(f"Colunas disponíveis para EDA: {list(df.columns)}") # [cite: 1454] Mesmo log

        # Gera contagem APÓS garantir que 'species' existe
        df_count = df['species'].value_counts().reset_index()
        # Renomeia explicitamente para evitar problemas se o nome do index mudar
        df_count.columns = ['species', 'count']

        # Chama as funções auxiliares (que agora retornam dicts)
        chart_data = {
            "chart1": _get_apex_scatter(df, 'sepal_length', 'sepal_width', 'Comprimento vs Largura da Sépala'),
            "chart2": _get_apex_scatter(df, 'petal_length', 'petal_width', 'Comprimento vs Largura da Pétala'),
            "chart3": _get_apex_bar(df_count, 'species', 'count', 'Contagem por Espécie'),
            "chart4": _get_apex_boxplot(df, 'species', 'sepal_length', 'Distribuição do Comprimento da Sépala'),
            "chart5": _get_apex_histogram(df, 'petal_length', 'Distribuição do Comprimento da Pétala'),
            "chart6": _get_apex_histogram(df, 'sepal_width', 'Distribuição da Largura da Sépala')
        }

        # Log de sucesso/erro por gráfico (sem alterações)
        for i, (key, chart) in enumerate(chart_data.items(), 1):
            if isinstance(chart, dict) and 'error' in chart:
                logger.warning(f"Gráfico {i} ({key}) com erro: {chart.get('error')}")
            else:
                logger.info(f"Gráfico {i} ({key}) gerado com sucesso") # [cite: 1455] Mesmo log

        context = {"request": request, "charts": chart_data} # Passa os dicts diretamente
        # CORREÇÃO: Certifique-se que o nome do template está correto
        return templates.TemplateResponse("partials/_iris_analise_eda.html", context) # [cite: 720] Usa o nome correto

    except HTTPException as h_exc:
        raise h_exc # Re-raise HTTPExceptions
    except Exception as e:
        logger.exception(f"Erro GERAL ao gerar gráficos EDA: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro geral ao gerar gráficos EDA: {str(e)}</div>",
                            status_code=500)

# Rota do gráfico PCA
@router.get("/pca_chart", response_class=HTMLResponse)
async def get_pca_chart(request: Request):
    try:
        df = load_and_prepare_data() # Garante colunas corretas
        chart_data = _get_apex_pca_2d(df, 'PCA 2D - Visualização das Espécies') # Renomeado e agora retorna dict

        # logger.info(f"Dados PCA enviados para template: {chart_data}") # Log mais detalhado se necessário

        context = {"request": request, "chart_pca": chart_data} # Passa o dict diretamente
        if isinstance(chart_data, dict) and 'error' in chart_data:
            context["error_message"] = f"Não foi possível gerar o gráfico PCA: {chart_data.get('error')}"
            logger.warning(f"Erro detectado ao gerar PCA: {chart_data.get('error')}")

        # CORREÇÃO: Certifique-se que o nome do template está correto
        return templates.TemplateResponse("partials/_iris_analise_pca.html", context) # [cite: 690] Usa o nome correto
    except HTTPException as h_exc:
        raise h_exc # Re-raise HTTPExceptions
    except Exception as e:
        logger.exception(f"Erro GERAL ao gerar gráfico PCA: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro geral ao gerar gráfico PCA: {str(e)}</div>",
                             status_code=500)