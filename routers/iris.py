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

EXPECTED_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target', 'species']
ORIGINAL_COLUMNS_MAP = {
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width'
}

@lru_cache(maxsize=1)
def load_and_prepare_data() -> pd.DataFrame:
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
            renamed = False
            cols_to_rename = {k: v for k, v in ORIGINAL_COLUMNS_MAP.items() if k in df.columns}
            if cols_to_rename:
                df.rename(columns=cols_to_rename, inplace=True)
                logger.warning(f"Colunas renomeadas do formato antigo para o novo: {cols_to_rename}")
                renamed = True

            missing_expected = [col for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'] if col not in df.columns]
            if missing_expected:
                logger.error(f"CSV existente ({FILE_PATH}) não contém colunas esperadas: {missing_expected}. Exclua o arquivo e reinicie para recriá-lo.")
                raise HTTPException(status_code=500, detail=f"CSV corrompido ou formato inesperado: {FILE_PATH.name}")

            if renamed:
                try:
                    df.to_csv(FILE_PATH, index=False)
                    logger.info(f"CSV atualizado em {FILE_PATH} com colunas padronizadas.")
                except Exception as e:
                    logger.error(f"Não foi possível salvar o CSV corrigido em {FILE_PATH}: {e}")

        except Exception as e:
            logger.error(f"Erro ao ler ou processar o CSV {FILE_PATH}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo {FILE_PATH.name}.")

    if 'species' not in df.columns and 'target' in df.columns:
        target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        df['species'] = df['target'].map(target_names).fillna('unknown')
        logger.info("Coluna 'species' adicionada ao DataFrame.")
    elif 'species' not in df.columns:
         logger.error("Coluna 'target' não encontrada para criar 'species'.")
         raise HTTPException(status_code=500, detail="Dataset Iris inválido: falta coluna 'target'.")

    final_missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if final_missing:
         logger.error(f"Após o processamento, colunas esperadas ainda estão faltando: {final_missing}")
         raise HTTPException(status_code=500, detail="Erro interno ao preparar colunas do dataset.")

    if created_now:
        try:
            df.to_csv(FILE_PATH, index=False)
            logger.info(f"Dataset Iris salvo em {FILE_PATH} com colunas padronizadas.")
        except Exception as e:
            logger.error(f"Não foi possível salvar o novo CSV em {FILE_PATH}: {e}")

    logger.info(f"Dataset Iris pronto ({len(df)} linhas). Colunas: {list(df.columns)}")
    return df

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            cleaned_list = []
            for item in obj.tolist():
                if isinstance(item, (float, np.float_)) and (np.isnan(item) or np.isinf(item)):
                    cleaned_list.append(None)
                else:
                    cleaned_list.append(item)
            return cleaned_list
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        elif pd.isna(obj):
             return None
        try:
             return json.JSONEncoder.default(self, obj)
        except TypeError:
            logger.warning(f"NpEncoder: Tipo não serializável encontrado e convertido para string: {type(obj)}")
            return str(obj)

def _safe_json_dumps(data):
    """ Serializes data to JSON string handling numpy types and NaN/Inf """
    try:
        return json.dumps(data, cls=NpEncoder, allow_nan=False)
    except Exception as e:
        logger.error(f"Erro final durante _safe_json_dumps: {e}", exc_info=True)
        return json.dumps({"error": f"Erro de serialização: {str(e)}"})

def _get_apex_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    try:
        series = []
        if x_col not in df.columns or y_col not in df.columns or 'species' not in df.columns:
            logger.error(f"Colunas ausentes para scatter plot '{title}': X='{x_col}', Y='{y_col}', Group='species'. Colunas disponíveis: {list(df.columns)}")
            return {'error': f'Colunas necessárias ausentes'}

        for species in df['species'].unique():
            species_df = df[df['species'] == species]
            x_data = [float(round(x, 2)) if pd.notna(x) else None for x in species_df[x_col]]
            y_data = [float(round(y, 2)) if pd.notna(y) else None for y in species_df[y_col]]
            data = [[x, y] for x, y in zip(x_data, y_data) if x is not None and y is not None]
            series.append({'name': str(species), 'data': data})

        chart_data = {
            'series': series,
            'chart': {'type': 'scatter', 'height': 350, 'zoom': {'enabled': True, 'type': 'xy'}},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'title': {'text': x_col}, 'tickAmount': 10},
            'yaxis': {'title': {'text': y_col}},
            'legend': {'position': 'top'}
        }
        return chart_data
    except Exception as e:
        logger.error(f"Erro ao criar scatter plot {title}: {e}", exc_info=True)
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_bar(df_agg: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    try:
        if x_col not in df_agg.columns or y_col not in df_agg.columns:
            logger.error(f"Colunas ausentes para bar chart '{title}': X='{x_col}', Y='{y_col}'. Colunas disponíveis: {list(df_agg.columns)}")
            return {'error': f'Colunas necessárias ausentes'}
        data_points = [float(v) if pd.notna(v) else None for v in df_agg[y_col].tolist()]
        chart_data = {
            'series': [{'name': y_col, 'data': data_points}],
            'chart': {'type': 'bar', 'height': 350},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'categories': [str(c) for c in df_agg[x_col].tolist()]},
            'plotOptions': {'bar': {'distributed': True}},
            'legend': {'show': False}
        }
        return chart_data
    except Exception as e:
        logger.error(f"Erro ao criar bar chart {title}: {e}", exc_info=True)
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> dict:
    try:
        if x_col not in df.columns or y_col not in df.columns:
            logger.error(f"Colunas ausentes para boxplot '{title}': X='{x_col}', Y='{y_col}'. Colunas disponíveis: {list(df.columns)}")
            return {'error': f'Colunas necessárias ausentes'}

        series_data = []
        for group_name in sorted(df[x_col].unique()):
            values = df[df[x_col] == group_name][y_col].dropna()
            if len(values) >= 5:
                stats = values.describe()
                y_values = [
                    float(round(stats.get('min', 0), 2)) if pd.notna(stats.get('min')) else None,
                    float(round(stats.get('25%', 0), 2)) if pd.notna(stats.get('25%')) else None,
                    float(round(stats.get('50%', 0), 2)) if pd.notna(stats.get('50%')) else None,
                    float(round(stats.get('75%', 0), 2)) if pd.notna(stats.get('75%')) else None,
                    float(round(stats.get('max', 0), 2)) if pd.notna(stats.get('max')) else None
                ]
                if any(v is None for v in y_values):
                    logger.warning(f"Dados insuficientes ou inválidos para boxplot '{title}' no grupo '{group_name}'. Pulando.")
                    continue
                series_data.append({'x': str(group_name), 'y': y_values})
            else:
                logger.warning(f"Grupo '{group_name}' tem menos de 5 pontos válidos para boxplot '{title}'. Pulando.")

        if not series_data:
            logger.error(f"Nenhum grupo válido encontrado para boxplot '{title}'.")
            return {'error': 'Dados insuficientes para gerar boxplot.'}

        chart_data = {
            'series': [{'name': y_col, 'type': 'boxPlot', 'data': series_data}],
            'chart': {'type': 'boxPlot', 'height': 350},
            'title': {'text': title, 'align': 'left'}
        }
        return chart_data
    except Exception as e:
        logger.error(f"Erro ao criar boxplot {title}: {e}", exc_info=True)
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_histogram(df: pd.DataFrame, x_col: str, title: str) -> dict:
    try:
        if x_col not in df.columns:
            logger.error(f"Coluna ausente para histograma '{title}': X='{x_col}'. Colunas disponíveis: {list(df.columns)}")
            return {'error': f'Coluna necessária ausente'}

        values = df[x_col].dropna()
        if len(values) == 0: return {'error': 'Sem dados para histograma'}

        try:
            counts, bins = np.histogram(values, bins=10)
        except ValueError as ve:
            logger.warning(f"Não foi possível calcular histograma para '{title}' (provavelmente valores constantes): {ve}")
            min_val, max_val = values.min(), values.max()
            if min_val == max_val:
                bins = np.linspace(min_val - 0.5, max_val + 0.5, 3)
                counts, bins = np.histogram(values, bins=bins)
            else:
                return {'error': f'Erro ao calcular bins: {ve}'}

        bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(counts))]

        chart_data = {
            'series': [{'name': 'Frequência', 'data': [int(c) for c in counts.tolist()]}],
            'chart': {'type': 'bar', 'height': 350},
            'title': {'text': title, 'align': 'left'},
            'xaxis': {'categories': bin_labels, 'title': {'text': x_col}},
            'yaxis': {'title': {'text': 'Frequência'}}, 'legend': {'show': False}
        }
        return chart_data
    except Exception as e:
        logger.error(f"Erro ao criar histograma {title}: {e}", exc_info=True)
        return {'error': f'Erro ao gerar gráfico: {str(e)}'}

def _get_apex_pca_2d(df: pd.DataFrame, title: str) -> dict:
    try:
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        missing_numeric = [col for col in numeric_cols if col not in df.columns]
        if missing_numeric:
            logger.error(f"Colunas numéricas ausentes para PCA: {missing_numeric}. Colunas disponíveis: {list(df.columns)}")
            return {'error': f'Colunas para PCA ausentes'}
        if 'species' not in df.columns:
            logger.error(f"Coluna 'species' ausente para PCA. Colunas disponíveis: {list(df.columns)}")
            return {'error': f"Coluna 'species' ausente"}

        X = df[numeric_cols].dropna()
        y_species = df.loc[X.index, 'species']
        if len(X) < 2: return {'error': 'Dados insuficientes para PCA'}

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        var_explained = f"{title} (Var. PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})"
        pca_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
        pca_df['species'] = y_species.reset_index(drop=True)

        series = []
        for species in pca_df['species'].unique():
            species_df = pca_df[pca_df['species'] == species]
            x_data = [float(round(x, 3)) if pd.notna(x) else None for x in species_df['PC1']]
            y_data = [float(round(y, 3)) if pd.notna(y) else None for y in species_df['PC2']]
            data = [[x, y] for x, y in zip(x_data, y_data) if x is not None and y is not None]
            series.append({'name': str(species), 'data': data})

        chart_data = {
            'series': series, 'chart': {'type': 'scatter', 'height': 450, 'zoom': {'enabled': True, 'type': 'xy'}},
            'title': {'text': var_explained, 'align': 'left'}, 'xaxis': {'title': {'text': 'PC1'}},
            'yaxis': {'title': {'text': 'PC2'}}, 'legend': {'position': 'top'}, 'markers': {'size': 5}
        }
        logger.info(f"PCA 2D gerado com sucesso: {len(X)} amostras, variância: {pca.explained_variance_ratio_}")
        return chart_data
    except Exception as e:
        logger.error(f"Erro ao criar PCA 2D: {e}", exc_info=True)
        return {'error': f'Erro ao gerar PCA: {str(e)}'}

@router.get("/", response_class=HTMLResponse)
async def get_iris_page(request: Request):
    settings = get_current_settings()
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("iris.html", context)

@router.get("/tabela", response_class=HTMLResponse)
async def get_tabela_iris(request: Request, page: int = Query(1, ge=1)):
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
        data = df_page.where(pd.notna(df_page), None).to_dict('records')

        context = {
            "request": request, "headers": headers, "data": data,
            "page": current_page, "total_pages": total_pages,
            "start_idx": start_idx + 1 if total_rows > 0 else 0,
            "end_idx": end_idx, "total_rows": total_rows,
        }
        return templates.TemplateResponse("partials/_iris_tabela.html", context)
    except HTTPException as h_exc: raise h_exc
    except Exception as e:
        logger.error(f"Erro ao carregar tabela Iris: {e}", exc_info=True)
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar dados: {str(e)}</div>", status_code=500)

@router.get("/eda_charts", response_class=HTMLResponse)
async def get_eda_charts(request: Request):
    try:
        df = load_and_prepare_data()
        logger.info(f"Colunas disponíveis para EDA: {list(df.columns)}")

        df_count = df['species'].value_counts().reset_index()
        df_count.columns = ['species', 'count']

        raw_chart_data = {
            "#chart-eda-1": _get_apex_scatter(df, 'sepal_length', 'sepal_width', 'Comprimento vs Largura da Sépala'),
            "#chart-eda-2": _get_apex_scatter(df, 'petal_length', 'petal_width', 'Comprimento vs Largura da Pétala'),
            "#chart-eda-3": _get_apex_bar(df_count, 'species', 'count', 'Contagem por Espécie'),
            "#chart-eda-4": _get_apex_boxplot(df, 'species', 'sepal_length', 'Distribuição do Comprimento da Sépala'),
            "#chart-eda-5": _get_apex_histogram(df, 'petal_length', 'Distribuição do Comprimento da Pétala'),
            "#chart-eda-6": _get_apex_histogram(df, 'sepal_width', 'Distribuição da Largura da Sépala')
        }

        charts_json_string = _safe_json_dumps(raw_chart_data)

        if '"error":' in charts_json_string:
             logger.warning(f"Um ou mais gráficos EDA podem ter falhado na geração/serialização. JSON: {charts_json_string[:500]}...")
        else:
             logger.info("Todos os gráficos EDA serializados com sucesso.")

        context = {"request": request, "charts_json": charts_json_string}
        return templates.TemplateResponse("partials/_iris_analise_eda.html", context)

    except HTTPException as h_exc: raise h_exc
    except Exception as e:
        logger.exception(f"Erro GERAL ao gerar gráficos EDA: {e}")
        error_data = {f"#chart-eda-{i}": {"error": f"Erro geral: {str(e)}"} for i in range(1, 7)}
        context = {"request": request, "charts_json": _safe_json_dumps(error_data)}
        return templates.TemplateResponse("partials/_iris_analise_eda.html", context, status_code=500)

@router.get("/pca_chart", response_class=HTMLResponse)
async def get_pca_chart(request: Request):
    try:
        df = load_and_prepare_data()
        raw_chart_data = _get_apex_pca_2d(df, 'PCA 2D - Visualização das Espécies')

        chart_pca_json_string = _safe_json_dumps(raw_chart_data)
        error_message_for_template = None

        if isinstance(raw_chart_data, dict) and 'error' in raw_chart_data:
             error_message_for_template = raw_chart_data.get('error')
             logger.warning(f"Erro detectado ao gerar PCA: {error_message_for_template}")
        elif '"error":' in chart_pca_json_string:
             try:
                 error_data = json.loads(chart_pca_json_string)
                 error_message_for_template = error_data.get("error", "Erro desconhecido na serialização")
             except:
                 error_message_for_template = "Erro complexo na serialização JSON do PCA."
             logger.error(f"Erro durante serialização JSON do PCA: {error_message_for_template}")

        context = {
            "request": request,
            "chart_pca_json": chart_pca_json_string,
            "error_message": error_message_for_template
        }

        return templates.TemplateResponse("partials/_iris_analise_pca.html", context)
    except HTTPException as h_exc: raise h_exc
    except Exception as e:
        logger.exception(f"Erro GERAL ao gerar gráfico PCA: {e}")
        error_data = {"error": f"Erro geral: {str(e)}"}
        context = {"request": request, "chart_pca_json": _safe_json_dumps(error_data), "error_message": f"Erro geral: {str(e)}"}
        return templates.TemplateResponse("partials/_iris_analise_pca.html", context, status_code=500)