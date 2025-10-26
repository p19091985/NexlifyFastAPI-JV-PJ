                                                                                       

import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris, fetch_covtype
from sklearn.decomposition import PCA

from fastapi import APIRouter, Request, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import auth as app_auth
from utils.settings import get_current_settings

from utils.analise_covertype_logic import (
    get_statistical_summary,
    run_classification_models,
    export_covertype_to_csv
)
from utils.analise_iris_logic import (
    export_iris_to_csv,
    _prepare_species_column,
    get_interactive_eda_figs,
    get_interactive_pca_fig
)

router = APIRouter(
    tags=["Análise"],
                                                        
    dependencies=[Depends(app_auth.get_current_user)]
)
templates = Jinja2Templates(directory="templates")

def load_data_from_csv(export_func, dataset_name):
    """Função auxiliar para carregar dados de CSVs cacheados."""
    csv_path = export_func()
                                                                 
    full_csv_path = Path(csv_path) if Path(csv_path).is_absolute() else Path() / csv_path

    if not full_csv_path.exists():
        raise HTTPException(status_code=500, detail=f"Falha ao carregar dataset '{dataset_name}'. Arquivo não encontrado em '{full_csv_path}'. Verifique se a função export o criou.")
    try:
        df = pd.read_csv(full_csv_path)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo CSV '{full_csv_path}': {e}")

@router.get("/iris", response_class=HTMLResponse)
async def get_iris_page(request: Request):
    """Renderiza a página principal de Análise Iris."""
    settings = get_current_settings()                   
    context = {"request": request, "settings": settings}                       
    return templates.TemplateResponse("iris.html", context)

@router.get("/iris/tabela", response_class=HTMLResponse)
async def get_iris_table(request: Request, page: int = Query(1, ge=1)):
    """Carrega os dados do Iris e retorna a tabela paginada (HTMX)."""
    try:
        df = load_data_from_csv(export_iris_to_csv, "Iris")

        PAGE_SIZE = 30
        total_rows = len(df)  
        total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)                              
        page = min(page, total_pages)                                          

        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)

        headers_list = [str(col) for col in df.columns.tolist()]
                                                                                              
        table_data = df.iloc[start_idx:end_idx].to_dict('records')

        return templates.TemplateResponse("partials/_iris_tabela.html", {
            "request": request,
            "data": table_data,  
            "headers": headers_list,                         
            "page": page,
            "total_pages": total_pages,
            "total_rows": total_rows,
            "start_idx": start_idx + 1 if total_rows > 0 else 0,                                   
            "end_idx": end_idx
        })
    except HTTPException as h_exc:                         
         raise h_exc
    except Exception as e:
        print(f"Erro ao gerar tabela Iris: {e}")
                                                                    
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar dados da tabela: {e}</div>", status_code=500)

@router.get("/iris/analise", response_class=HTMLResponse)
async def get_iris_analysis(request: Request):  
    """Executa a análise EDA e PCA do Iris e retorna os gráficos (HTMX)."""
    try:
        df = load_data_from_csv(export_iris_to_csv, "Iris")

        eda_figs = get_interactive_eda_figs(df)  
                         
        pca_fig = get_interactive_pca_fig(df)

        plot_jsons = [fig.to_json() for fig in eda_figs if fig]                      
        pca_json = pca_fig.to_json() if pca_fig else None  

        return templates.TemplateResponse("partials/_iris_analise.html", {
            "request": request,
            "plot_jsons": plot_jsons,
            "pca_json": pca_json
        })
    except HTTPException as h_exc:                         
         raise h_exc
    except Exception as e:
        print(f"Erro ao gerar análise Iris: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao executar análise: {e}</div>", status_code=500)

@router.get("/covertype", response_class=HTMLResponse)
async def get_covertype_page(request: Request):
    """Renderiza a página principal de Análise Covertype."""
    settings = get_current_settings()                   
    context = {"request": request, "settings": settings}                       
    return templates.TemplateResponse("covertype.html", context)

@router.get("/covertype/tabela", response_class=HTMLResponse)
async def get_covertype_table(request: Request, page: int = Query(1, ge=1)):
    """Carrega os dados do Covertype e retorna a tabela paginada (HTMX)."""
    try:
        df = load_data_from_csv(export_covertype_to_csv, "Covertype")

        PAGE_SIZE = 30
        total_rows = len(df)  
        total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)                              
        page = min(page, total_pages)                 

        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)

        headers_list = [str(col) for col in df.columns.tolist()]                  
        table_data = df.iloc[start_idx:end_idx].to_dict('records')  

        return templates.TemplateResponse("partials/_covertype_tabela.html", {
            "request": request,
            "data": table_data,  
            "headers": headers_list,  
            "page": page,  
            "total_pages": total_pages,
            "total_rows": total_rows,
            "start_idx": start_idx + 1 if total_rows > 0 else 0,                 
            "end_idx": end_idx
        })
    except HTTPException as h_exc:                         
         raise h_exc
    except Exception as e:
        print(f"Erro ao gerar tabela Covertype: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar dados da tabela: {e}</div>", status_code=500)

@router.get("/covertype/analise", response_class=HTMLResponse)
async def get_covertype_analysis(request: Request):  
    """Executa a análise estatística e de classificação do Covertype (HTMX)."""
    try:
        df = load_data_from_csv(export_covertype_to_csv, "Covertype")

        info_df, describe_df, target_dist = get_statistical_summary(df)

        results_df = run_classification_models(df)

        context = {  
            "request": request,  
            "info_html": info_df.to_html(classes="table table-sm table-striped table-bordered small", border=0, index=False),  
            "describe_html": describe_df.to_html(classes="table table-sm table-striped table-bordered small", border=0),  
            "target_dist_html": target_dist.to_html(classes="table table-sm table-striped table-bordered small", border=0, header=False),                       
            "results_html": results_df.to_html(classes="table table-sm table-striped table-hover table-bordered small", border=0,
                                               index=False, float_format="%.4f")  
        }  

        return templates.TemplateResponse("partials/_covertype_analise.html", context)
    except HTTPException as h_exc:                         
         raise h_exc
    except Exception as e:
        print(f"Erro ao gerar análise Covertype: {e}")
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao executar análise: {e}</div>", status_code=500)  