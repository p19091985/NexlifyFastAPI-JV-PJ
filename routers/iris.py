                                              

import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
import plotly.express as px
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
    Carrega o dataset Iris. Se não existir, baixa, SALVA COM NOMES LIMPOS e carrega.
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

            df.columns = [col.replace(' (cm)', '').replace(' ', '_').lower() for col in df.columns]

            df.to_csv(FILE_PATH, index=False)
            logger.info(f"Dataset Iris salvo em {FILE_PATH} com colunas LIMPAS.")
        except Exception as e:
            logger.error(f"Falha crítica: Não foi possível baixar ou salvar o dataset Iris: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Erro ao baixar ou salvar o dataset Iris.")

    try:
        df = pd.read_csv(FILE_PATH)
        logger.info(f"Dataset Iris carregado de {FILE_PATH} ({len(df)} linhas).")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Erro: O arquivo CSV {FILE_PATH} está vazio.")
        raise HTTPException(status_code=500, detail=f"Arquivo {FILE_PATH.name} está vazio.")
    except Exception as e:
        logger.error(f"Erro ao ler o CSV {FILE_PATH}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo {FILE_PATH.name}.")

def _prepare_species_column(df):
    df = df.copy()
    if 'species' in df.columns: return df
    if 'target' in df.columns:
        target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        df['species'] = df['target'].map(target_names)
        df['species'] = df['species'].fillna('unknown')
        logger.debug("Coluna 'species' criada a partir de 'target'.")
    else:
        logger.warning("Coluna 'target' não encontrada para criar 'species'.")
        df['species'] = 'unknown'
    return df

def _df_to_tuple_for_cache(df):
    return tuple([tuple(df.columns)] + list(df.itertuples(index=False, name=None)))

def _tuple_to_df_from_cache(df_tuple):
    if not df_tuple or len(df_tuple) < 1: return pd.DataFrame()
    return pd.DataFrame(list(df_tuple[1:]), columns=list(df_tuple[0]))

@lru_cache(maxsize=2)
def get_cached_eda_figs_json(df_tuple):
    logger.debug("Gerando gráficos EDA (cache miss ou nova chamada)...")
    df = _tuple_to_df_from_cache(df_tuple)
    if df.empty: return ["error"]
    try:
        df = _prepare_species_column(df)
        figs = []

        figs.append(px.scatter(df, x='petal_length', y='petal_width',
                               color='species', title='Comprimento vs. Largura da Pétala'))
        figs.append(px.scatter(df, x='sepal_length', y='sepal_width',
                               color='species', title='Comprimento vs. Largura da Sépala'))

        count_df = df['species'].value_counts().reset_index()
        count_df.columns = ['species', 'count']
        figs.append(px.bar(count_df, x='species', y='count', color='species', title='Contagem por Espécie'))

        figs.append(px.box(df, x='species', y='sepal_length',
                           color='species', title='Box Plot Comprimento Sépala'))
        figs.append(px.violin(df, x='species', y='petal_width',
                              color='species', title='Violin Plot Largura Pétala'))
        figs.append(px.histogram(df, x='petal_length',
                                 color='species', title='Distribuição Comprimento Pétala'))

        dict_figs = [json.loads(fig.to_json()) for fig in figs]

        logger.debug("Gráficos EDA gerados com sucesso.")
        return dict_figs
    except Exception as e:
        logger.error(f"Erro ao gerar gráficos EDA: {e}", exc_info=True)
        return ["error"]

@lru_cache(maxsize=2)
def get_cached_pca_fig_json(df_tuple):
    logger.debug("Gerando gráfico PCA (cache miss ou nova chamada)...")
    df = _tuple_to_df_from_cache(df_tuple)
    if df.empty: return "error"
    try:
        df = _prepare_species_column(df)
        y_species = df['species']

        if 'target' in df.columns:
            X = df.drop(columns=['target', 'species'], errors='ignore')
        else:
            X = df.drop(columns=['species'], errors='ignore')

        X_numeric = X.select_dtypes(include=[np.number])

        if X_numeric.shape[1] < 3:
            logger.warning(f"PCA 3D requer >= 3 colunas numéricas, encontrado {X_numeric.shape[1]}.")
            return None

        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X_numeric)

        pca_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3'])
        pca_df['species'] = y_species

        fig_pca = px.scatter_3d(
            pca_df,
            x='PC1', y='PC2', z='PC3',
            color='species', title='Visualização PCA 3D das Features Iris'
        )

        dict_pca = json.loads(fig_pca.to_json())

        logger.debug("Gráfico PCA gerado com sucesso.")
        return dict_pca
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico PCA: {e}", exc_info=True)
        return "error"

@router.get("/", response_class=HTMLResponse)
async def get_iris_page(request: Request):
    """Renderiza a página principal do painel Iris."""
    settings = get_current_settings()
    try:
        load_and_prepare_data()               
    except Exception:
        pass
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
        start_idx_0based = (current_page - 1) * PAGE_SIZE
        end_idx_0based = min(start_idx_0based + PAGE_SIZE, total_rows)
        df_page = df.iloc[start_idx_0based:end_idx_0based]
        headers = [str(col) for col in df.columns.tolist()]
        data = df_page.to_dict('records')
        context = {
            "request": request, "headers": headers, "data": data,
            "page": current_page, "total_pages": total_pages,
            "start_idx": start_idx_0based + 1 if total_rows > 0 else 0,
            "end_idx": end_idx_0based, "total_rows": total_rows,
            "config": config
        }

        return templates.TemplateResponse("partials/_iris_tabela.html", context)

    except HTTPException as http_err:
        logger.error(f"Erro HTTP ao carregar tabela Iris: {http_err.detail}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar dados: {http_err.detail}</div>",
                            status_code=http_err.status_code)
    except Exception as e:
        logger.exception("Erro inesperado ao gerar tabela Iris.")
        return HTMLResponse(
            f"<div class='alert alert-danger'>Erro inesperado ao gerar a tabela. Consulte os logs.</div>",
            status_code=500)

@router.get("/analise", response_class=HTMLResponse)
async def get_analise_iris(request: Request):
    """Gera gráficos e retorna parcial."""
    try:
        df = load_and_prepare_data()
        df_tuple = _df_to_tuple_for_cache(df)
        logger.info("Gerando ou buscando do cache os gráficos EDA e PCA...")
        plot_jsons_or_error = get_cached_eda_figs_json(df_tuple)
        pca_json_or_error = get_cached_pca_fig_json(df_tuple)
        error_eda = isinstance(plot_jsons_or_error, list) and "error" in plot_jsons_or_error
        error_pca = pca_json_or_error == "error"
        context = {
            "request": request,
            "plot_jsons": plot_jsons_or_error,
            "pca_json": pca_json_or_error,
            "config": config
        }
        status_code = 500 if error_eda or error_pca else 200
        if status_code == 500:
            logger.error(f"Erro detectado na geração dos JSONs dos gráficos (EDA: {error_eda}, PCA: {error_pca}).")
        logger.info("Renderizando parcial de análise Iris.")

        return templates.TemplateResponse("partials/_iris_analise.html", context, status_code=status_code)

    except HTTPException as http_err:
        logger.error(f"Erro HTTP ao preparar análise Iris: {http_err.detail}")
        return HTMLResponse(
            f"<div class='alert alert-danger'>Erro ao carregar dados para análise: {http_err.detail}</div>",
            status_code=http_err.status_code)
    except Exception as e:
        logger.exception(f"Erro inesperado ao gerar análise Iris: {e}")                          
        return HTMLResponse(
            f"<div class='alert alert-danger'>Erro inesperado ao gerar a análise. Consulte os logs.</div>",
            status_code=500)