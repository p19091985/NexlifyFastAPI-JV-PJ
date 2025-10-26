                                                                           

import pandas as pd
import numpy as np
import sys
import os
import logging
from pathlib import Path
from functools import lru_cache

from fastapi import APIRouter, Request, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import auth as app_auth
import config                 
from utils.settings import get_current_settings

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/covertype",
    tags=["Análise Covertype"],
    dependencies=[Depends(app_auth.get_current_user)]
)
templates = Jinja2Templates(directory="templates")

CSV_DIR = Path("csv")
FILE_PATH = CSV_DIR / 'covertype_dataset.csv'

def export_covertype_to_csv() -> str | None:
    if FILE_PATH.exists():
        logger.debug(f"Arquivo CSV já existe em: {FILE_PATH}")
        return str(FILE_PATH)
    if not CSV_DIR.exists():
        logger.info(f"Criando diretório: {CSV_DIR}")
        try:
            CSV_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Erro: Não foi possível criar o diretório '{CSV_DIR}': {e}")
            return None
    try:
        logger.info("Buscando o dataset Covertype da nuvem...")
        covtype = fetch_covtype(as_frame=True)
        df = covtype.frame
        logger.info("Download concluído. Salvando em CSV...")
        df.to_csv(FILE_PATH, index=False)
        logger.info(f"Dataset salvo em {FILE_PATH}")
        return str(FILE_PATH)
    except Exception as e:
        logger.error(f"Erro: Não foi possível baixar ou salvar o dataset Covertype: {e}")
        if FILE_PATH.exists():
            try: FILE_PATH.unlink()
            except OSError: pass
        return None

def get_statistical_summary(df: pd.DataFrame) -> tuple:
    logger.debug("Gerando resumo estatístico...")
    info_df = pd.DataFrame({
        "Coluna": df.columns, "Tipo (Dtype)": df.dtypes.astype(str),
        "Valores Não-Nulos": df.count().values, "Valores Nulos": df.isnull().sum().values,
        "Valores Únicos": df.nunique().values
    }).reset_index(drop=True)
    describe_df = df.describe()
    target_col = 'Cover_Type'
    target_distribution_df = pd.DataFrame(columns=["count"])
    if target_col in df.columns:
        target_distribution_df = df[target_col].value_counts().sort_index().to_frame()
        target_distribution_df.columns = ['Contagem']
        target_distribution_df.index.name = target_col
    else:
        logger.warning(f"Coluna alvo '{target_col}' não encontrada.")
    logger.debug("Resumo estatístico gerado.")
    return info_df, describe_df, target_distribution_df

def get_balanced_sample(df: pd.DataFrame, n_per_class: int = 1000, target_col: str = 'Cover_Type') -> pd.DataFrame:
    if target_col not in df.columns:
        logger.error(f"Coluna alvo '{target_col}' não encontrada.")
        return pd.DataFrame()
    min_samples_in_class = df[target_col].value_counts().min()
    actual_n_per_class = min(n_per_class, min_samples_in_class)
    if actual_n_per_class < n_per_class:
        logger.warning(f"Ajustando amostragem para {actual_n_per_class} por classe.")
    elif actual_n_per_class <= 0:
        logger.error("Amostras insuficientes.")
        return pd.DataFrame()
    logger.info(f"Criando amostra balanceada com {actual_n_per_class} exemplos por classe...")
    try:
                                                                      
        balanced_sample_df = df.groupby(target_col, group_keys=False).apply(
             lambda x: x.sample(n=actual_n_per_class, random_state=42)
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Amostra balanceada criada com {len(balanced_sample_df)} registros.")
        return balanced_sample_df
    except Exception as e:
         logger.error(f"Erro durante a amostragem balanceada: {e}", exc_info=True)                      
         return pd.DataFrame()                                   

def run_classification_models(df: pd.DataFrame) -> pd.DataFrame:
    target_col = 'Cover_Type'
    if target_col not in df.columns:
        logger.error("Coluna 'Cover_Type' não encontrada.")
        return pd.DataFrame()

    try:
        df_sample = get_balanced_sample(df, n_per_class=1000, target_col=target_col)
        if df_sample.empty:
            logger.error("Amostragem balanceada retornou um DataFrame vazio. Verifique os logs anteriores.")
                                                 
            return pd.DataFrame(columns=["Classificador", "Acurácia", "Precisão", "Recall", "F1-Score"])
    except Exception as e:
        logger.error(f"Exceção não tratada durante a amostragem: {e}", exc_info=True)
        return pd.DataFrame(columns=["Classificador", "Acurácia", "Precisão", "Recall", "F1-Score"])

    logger.info(f"Iniciando treinamento de modelos em {len(df_sample)} amostras...")
    X = df_sample.drop(target_col, axis=1)
    y = df_sample[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced'),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "KNN (K=5)": KNeighborsClassifier(n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Análise Discriminante Linear": LinearDiscriminantAnalysis(),
        "MLP (Rede Neural Simples)": MLPClassifier(max_iter=300, random_state=42, early_stopping=True, hidden_layer_sizes=(50,), alpha=0.001)
    }
    results = []
    total_models = len(models)
    logger.info(f"Treinando {total_models} modelos...")
    for i, (name, model) in enumerate(models.items()):
        logger.info(f"  ({i + 1}/{total_models}) Treinando: {name}...")
        y_pred = None
        try:
            if name in ["Regressão Logística", "KNN (K=5)", "MLP (Rede Neural Simples)", "Análise Discriminante Linear"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        except Exception as e:
            logger.warning(f"  Falha ao treinar o modelo '{name}': {e}")
            accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan
                                                                                        
        if y_pred is not None:
             try:
                 accuracy = accuracy_score(y_test, y_pred)
                 precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                 recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                 f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                 logger.info(f"  {name} - F1-Score (Weighted): {f1:.4f}")
             except Exception as metric_e:
                 logger.warning(f"  Erro ao calcular métricas para '{name}': {metric_e}")
                 accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan
        else:                                    
             accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan

        results.append({
            "Classificador": name, "Acurácia": accuracy, "Precisão": precision,
            "Recall": recall, "F1-Score": f1
        })

    logger.info("Treinamento concluído.")
                                                                                 
    if not results:
        logger.error("Nenhum modelo foi treinado com sucesso.")
        return pd.DataFrame(columns=["Classificador", "Acurácia", "Precisão", "Recall", "F1-Score"])

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return results_df

@lru_cache()
def load_data_with_cache() -> pd.DataFrame:
    logger.info("Tentando carregar dados do Covertype (com cache)...")
    csv_path_str = export_covertype_to_csv()
    if not csv_path_str:
        logger.error("Falha ao obter o caminho do CSV.")
        return pd.DataFrame()
    full_csv_path = Path(csv_path_str)
    if not full_csv_path.exists():
         logger.error(f"Arquivo CSV não encontrado em '{full_csv_path}'.")
         return pd.DataFrame()
    try:
        df = pd.read_csv(full_csv_path)
        logger.info(f"Dataset Covertype carregado ({len(df)} linhas).")
        if 'Cover_Type' not in df.columns: logger.warning("Coluna 'Cover_Type' não encontrada.")
        return df
    except Exception as e:
        logger.error(f"Erro ao ler CSV '{full_csv_path}': {e}")
        return pd.DataFrame()

def _df_to_tuple_for_cache(df):
    return tuple([tuple(df.columns)] + list(df.itertuples(index=False, name=None)))

def _tuple_to_df_from_cache(df_tuple):
    if not df_tuple or len(df_tuple) < 1: return pd.DataFrame()
    return pd.DataFrame(list(df_tuple[1:]), columns=list(df_tuple[0]))

@lru_cache(maxsize=2)
def get_cached_stats(df_tuple):
    logger.debug("Executando get_statistical_summary...")
    df = _tuple_to_df_from_cache(df_tuple)
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return get_statistical_summary(df)

@lru_cache(maxsize=2)
def get_cached_models(df_tuple):
    logger.debug("Executando run_classification_models...")
    df = _tuple_to_df_from_cache(df_tuple)
                                                                                                        
    if df.empty:
        logger.error("DataFrame de entrada para get_cached_models está vazio.")
        return pd.DataFrame(columns=["Classificador", "Acurácia", "Precisão", "Recall", "F1-Score"])
    return run_classification_models(df)

@router.get("/", response_class=HTMLResponse)
async def get_covertype_page(request: Request):
    """Renderiza a página principal."""
    settings = get_current_settings()
    try: load_data_with_cache()              
    except Exception: pass
                                 
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("covertype.html", context)

@router.get("/tabela", response_class=HTMLResponse)
async def get_tabela_covertype(request: Request, page: int = Query(1, ge=1)):
    """Carrega dados e retorna tabela paginada."""
    try:
        df = load_data_with_cache()
        if df.empty:
            if not FILE_PATH.exists(): return HTMLResponse("<div class='alert alert-danger'>Erro: Arquivo CSV não encontrado.</div>", status_code=500)
            else: return HTMLResponse("<div class='alert alert-warning'>Dataset vazio ou erro ao carregar. Verifique os logs.</div>")                      

        PAGE_SIZE = 30
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
        return templates.TemplateResponse("partials/_covertype_tabela.html", context)
    except Exception as e:
        logger.exception("Erro ao carregar tabela.")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro inesperado ao carregar tabela. Verifique os logs.</div>", status_code=500)

@router.get("/analise", response_class=HTMLResponse)
async def get_analise_covertype(request: Request):
    """Executa análise e retorna resultados."""
    try:
        df = load_data_with_cache()
        if df.empty:
             if not FILE_PATH.exists(): return HTMLResponse("<div class='alert alert-danger'>Erro: Arquivo CSV não encontrado. Não é possível analisar.</div>", status_code=500)                      
             else: return HTMLResponse("<div class='alert alert-warning'>Dataset vazio ou erro ao carregar. Não é possível analisar. Verifique os logs.</div>")                      

        logger.info("Iniciando análise Covertype (resumo + modelos)...")
        df_tuple = _df_to_tuple_for_cache(df)
        info_df, describe_df, target_dist_df = get_cached_stats(df_tuple)
        results_df = get_cached_models(df_tuple)                                                                          
        logger.info("Análise concluída. Gerando HTML...")

        info_html = info_df.to_html(index=False, classes="table table-sm table-striped table-bordered small mb-0", border=0) if not info_df.empty else "<p class='text-danger m-3'>Erro ao gerar informações das colunas.</p>"
        describe_html = describe_df.to_html(classes="table table-sm table-striped table-bordered small mb-0", border=0, float_format='{:.2f}'.format) if not describe_df.empty else "<p class='text-danger m-3'>Erro ao gerar resumo descritivo.</p>"
        target_dist_html = target_dist_df.to_html(classes="table table-sm table-striped table-bordered small mb-0", border=0, header=True) if not target_dist_df.empty else "<p class='text-danger m-3'>Erro ao gerar distribuição do alvo.</p>"
        results_html = results_df.to_html(index=False, classes="table table-sm table-striped table-hover table-bordered small mb-0", border=0, float_format='{:.4f}'.format, na_rep='Falhou') if not results_df.empty else "<p class='text-danger m-3'>Erro ao treinar ou avaliar modelos. Verifique os logs para detalhes.</p>"                           

        context = {
            "request": request,
            "info_html": info_html, "describe_html": describe_html,
            "target_dist_html": target_dist_html, "results_html": results_html,
            "warning_sync": True,
            "config": config                                         
        }
        return templates.TemplateResponse("partials/_covertype_analise.html", context)
    except Exception as e:
        logger.exception("Erro GERAL durante a análise Covertype.")                     
                                                             
        return HTMLResponse(f"<div class='alert alert-danger'>Erro inesperado ao processar a análise completa. Consulte os logs do servidor para mais detalhes.</div>", status_code=500)