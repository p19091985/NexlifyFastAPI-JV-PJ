                                              

import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from functools import lru_cache
import json

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

@lru_cache(maxsize=1)
def load_and_prepare_data() -> pd.DataFrame:
    """
    Carrega o dataset Iris. Se não existir, baixa, limpa os nomes e salva em CSV.
    """
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
                                                                                      
            df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
                                       
            target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            df['species'] = df['target'].map(target_names)
            df.to_csv(FILE_PATH, index=False)
            logger.info(f"Dataset Iris salvo em {FILE_PATH} com colunas padronizadas.")
        except Exception as e:
            logger.error(f"Falha crítica: Não foi possível baixar ou salvar o dataset Iris: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Erro ao baixar ou salvar o dataset Iris.")

    try:
        df = pd.read_csv(FILE_PATH)
                                             
        if 'species' not in df.columns and 'target' in df.columns:
            target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            df['species'] = df['target'].map(target_names).fillna('unknown')
        logger.info(f"Dataset Iris carregado de {FILE_PATH} ({len(df)} linhas). Colunas: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Erro ao ler o CSV {FILE_PATH}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo {FILE_PATH.name}.")

def _safe_json_dumps(data):
    """Converte dados (incluindo numpy) para JSON seguro."""

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if pd.isna(obj):
                return None
            return super(NpEncoder, self).default(obj)

    return json.dumps(data, cls=NpEncoder)

def _get_apex_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    """Formata dados de scatter plot para ApexCharts."""
    try:
        series = []
        for species in df['species'].unique():
            species_df = df[df['species'] == species]
                            
            data = [[round(x, 2), round(y, 2)] for x, y in zip(species_df[x_col], species_df[y_col])]
            series.append({'name': species, 'data': data})

        chart_data = {
            'series': series,
            'chart': {'type': 'scatter', 'height': 350, 'zoom': {'enabled': True, 'type': 'xy'}},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'title': {'text': x_col}, 'tickAmount': 10},
            'yaxis': {'title': {'text': y_col}},
            'legend': {'position': 'top'}
        }
        return json.loads(_safe_json_dumps(chart_data))
    except Exception as e:
        logger.error(f"Erro ao criar scatter plot {title}: {e}")
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_bar(df_agg: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    """Formata dados de bar chart para ApexCharts."""
    try:
        chart_data = {
            'series': [{'name': y_col, 'data': df_agg[y_col].tolist()}],
            'chart': {'type': 'bar', 'height': 350},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'categories': df_agg[x_col].tolist()},
            'plotOptions': {'bar': {'distributed': True}},
            'legend': {'show': False}
        }
        return json.loads(_safe_json_dumps(chart_data))
    except Exception as e:
        logger.error(f"Erro ao criar bar chart {title}: {e}")
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    """Formata dados de box plot para ApexCharts."""
    try:
        series_data = []
        for group_name in sorted(df[x_col].unique()):
            values = df[df[x_col] == group_name][y_col].dropna()
            if len(values) > 0:
                stats = values.describe()
                series_data.append({
                    'x': group_name,
                    'y': [
                        round(stats.get('min', 0), 2),
                        round(stats.get('25%', 0), 2),
                        round(stats.get('50%', 0), 2),
                        round(stats.get('75%', 0), 2),
                        round(stats.get('max', 0), 2)
                    ]
                })

        chart_data = {
            'series': [{'name': y_col, 'type': 'boxPlot', 'data': series_data}],
            'chart': {'type': 'boxPlot', 'height': 350},
            'title': {'text': title, 'align': 'left'}
        }
        return json.loads(_safe_json_dumps(chart_data))
    except Exception as e:
        logger.error(f"Erro ao criar boxplot {title}: {e}")
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_histogram(df: pd.DataFrame, x_col: str, title: str) -> dict:
    """Formata dados de histograma para ApexCharts."""
    try:
        values = df[x_col].dropna()
        if len(values) == 0:
            return {'error': 'Sem dados para histograma'}

        counts, bins = np.histogram(values, bins=10)
        bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

        chart_data = {
            'series': [{'name': 'Frequência', 'data': counts.tolist()}],
            'chart': {'type': 'bar', 'height': 350},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'categories': bin_labels, 'title': {'text': x_col}},
            'yaxis': {'title': {'text': 'Frequência'}},
            'legend': {'show': False}
        }
        return json.loads(_safe_json_dumps(chart_data))
    except Exception as e:
        logger.error(f"Erro ao criar histograma {title}: {e}")
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_pca_3d(df: pd.DataFrame, title: str) -> dict:
    """Formata dados do PCA 2D (PC1 vs PC2) para ApexCharts. (Corrigido para 2D scatter)."""
    try:
                                                     
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
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
                                            
            data = [[round(x, 3), round(y, 3)] for x, y in zip(species_df['PC1'], species_df['PC2'])]
            series.append({'name': species, 'data': data})

        chart_data = {
            'series': series,
            'chart': {'type': 'scatter', 'height': 450, 'zoom': {'enabled': True, 'type': 'xy'}},
            'title': {'text': var_explained, 'align': 'left'},
            'xaxis': {'title': {'text': 'PC1'}},
            'yaxis': {'title': {'text': 'PC2'}},
            'legend': {'position': 'top'},
            'markers': {'size': 5}
        }
        logger.info(f"PCA 2D gerado com sucesso: {len(X)} amostras, variância: {pca.explained_variance_ratio_}")
        return json.loads(_safe_json_dumps(chart_data))
    except Exception as e:
        logger.error(f"Erro ao criar PCA 2D: {e}")
        return {'error': f'Erro ao gerar PCA: {str(e)}'}

@router.get("/", response_class=HTMLResponse)
async def get_iris_page(request: Request):
    """Renderiza a página principal do painel Iris."""
    settings = get_current_settings()
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("iris.html", context)

@router.get("/tabela", response_class=HTMLResponse)
async def get_tabela_iris(request: Request, page: int = Query(1, ge=1)):
    """Carrega dados e retorna tabela paginada."""
    try:
        df = load_and_prepare_data()
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
        return templates.TemplateResponse("partials/_iris_tabela.html", context)
    except Exception as e:
        logger.error(f"Erro ao carregar tabela: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar dados: {str(e)}</div>", status_code=500)

@router.get("/eda_charts", response_class=HTMLResponse)
async def get_eda_charts(request: Request):
    """Gera gráficos EDA e retorna parcial."""
    try:
        df = load_and_prepare_data()
        logger.info(f"Colunas disponíveis para EDA: {list(df.columns)}")

        df_count = df['species'].value_counts().reset_index()
        df_count.columns = ['species', 'count']

        chart_data = {
            "chart1": _get_apex_scatter(df, 'sepal_length', 'sepal_width', 'Comprimento vs Largura da Sépala'),
            "chart2": _get_apex_scatter(df, 'petal_length', 'petal_width', 'Comprimento vs Largura da Pétala'),
            "chart3": _get_apex_bar(df_count, 'species', 'count', 'Contagem por Espécie'),
            "chart4": _get_apex_boxplot(df, 'species', 'sepal_length', 'Distribuição do Comprimento da Sépala'),
            "chart5": _get_apex_histogram(df, 'petal_length', 'Distribuição do Comprimento da Pétala'),
            "chart6": _get_apex_histogram(df, 'sepal_width', 'Distribuição da Largura da Sépala')
        }

        for i, (key, chart) in enumerate(chart_data.items(), 1):
            if 'error' in str(chart):
                logger.warning(f"Gráfico {i} ({key}) com erro: {chart}")
            else:
                logger.info(f"Gráfico {i} ({key}) gerado com sucesso")

        context = {"request": request, "charts": chart_data}
        return templates.TemplateResponse("partials/_iris_analise_eda.html", context)

    except Exception as e:
        logger.exception(f"Erro ao gerar gráficos EDA: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao gerar gráficos EDA: {str(e)}</div>",
                            status_code=500)

@router.get("/pca_chart", response_class=HTMLResponse)
async def get_pca_chart(request: Request):
    """Gera gráfico PCA e retorna parcial."""
    try:
        df = load_and_prepare_data()
        chart_data = _get_apex_pca_3d(df, 'PCA 2D - Visualização das Espécies')

        logger.info(f"Dados PCA enviados: {chart_data}")                  

        context = {"request": request, "chart_pca": chart_data}
        if isinstance(chart_data, dict) and 'error' in str(chart_data):
            context["error_message"] = "Não foi possível gerar o gráfico PCA 2D"

        return templates.TemplateResponse("partials/_iris_analise_pca.html", context)
    except Exception as e:
        logger.exception(f"Erro ao gerar gráfico PCA: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao gerar gráfico PCA: {str(e)}</div>",
                            status_code=500)