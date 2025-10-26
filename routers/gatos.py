                                               

import pandas as pd
from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from persistencia.repository import GenericRepository
import auth as app_auth
import config                 
from utils.settings import get_current_settings

router = APIRouter(
    prefix="/gatos",                
    tags=["Gatos"],                  
    dependencies=[Depends(app_auth.get_current_user)]                                                           
)
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def get_gatos_page(request: Request):
    """Renderiza a página principal de Gatos."""
    settings = get_current_settings()
    gatos_list = []
    db_disabled = not config.DATABASE_ENABLED
    error_message = None

    if not db_disabled:
        try:
                                                   
            df = GenericRepository.read_table_to_dataframe('especie_gatos')
            gatos_list = df.to_dict('records')
        except Exception as e:
            error_message = f"Erro ao carregar gatos: {e}"
            print(error_message)

    context = {
        "request": request,
        "settings": settings,
        "gatos": gatos_list,
        "db_disabled": db_disabled,
        "error_message": error_message,
        "config": config                              
    }
    return templates.TemplateResponse("gatos.html", context)

@router.get("/novo", response_class=HTMLResponse)
async def get_gato_form(request: Request):
    """Retorna o formulário HTMX para adicionar um novo gato."""
                                            
    return templates.TemplateResponse("partials/_gato_form.html", {"request": request, "config": config})

@router.get("/{gato_id}/editar", response_class=HTMLResponse)
async def get_gato_edit_form(request: Request, gato_id: int):
    """Retorna o formulário HTMX preenchido para editar um gato."""
    try:
                                               
        df = GenericRepository.read_table_to_dataframe('especie_gatos', where_conditions={'id': gato_id})
        if df.empty:
            raise HTTPException(status_code=404, detail="Gato não encontrado")
        gato = df.to_dict('records')[0]
                                                
        return templates.TemplateResponse("partials/_gato_form.html", {"request": request, "gato": gato, "config": config})
    except HTTPException as http_exc:
        raise http_exc                
    except Exception as e:
                                                    
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar gato: {e}</div>", status_code=500)

@router.post("/", response_class=Response)                                   
async def create_gato(
    request: Request,
    nome_especie: str = Form(...),
    pais_origem: str = Form(""),
    temperamento: str = Form("")
):
    """Processa a criação de um novo gato (via HTMX)."""
    if not nome_especie.strip():
        return HTMLResponse("<div class='alert alert-danger'>O nome da espécie é obrigatório.</div>", status_code=400)

    try:
        df = pd.DataFrame([{'nome_especie': nome_especie.strip(),
                            'pais_origem': pais_origem.strip(),
                            'temperamento': temperamento.strip()}])
                                               
        GenericRepository.write_dataframe_to_table(df, 'especie_gatos')
                                           
        return Response(content="", headers={"HX-Refresh": "true"})
    except Exception as e:
        err_str = str(e).lower()
                                                                       
        if "unique constraint" in err_str or "duplicate entry" in err_str or "constraint failed" in err_str:
             msg = f"Erro: A espécie '{nome_especie}' já existe."
             status = 409           
        else:
            msg = f"Erro ao salvar: {e}"
            status = 500
        return HTMLResponse(f"<div class='alert alert-danger'>{msg}</div>", status_code=status)

@router.put("/{gato_id}", response_class=HTMLResponse)                             
async def update_gato(
        request: Request,
        gato_id: int,
        nome_especie: str = Form(...),
        pais_origem: str = Form(""),
        temperamento: str = Form("")
):
    """Processa a atualização de um gato existente (via HTMX)."""
    if not nome_especie.strip():
        return HTMLResponse("<div class='alert alert-danger'>O nome da espécie é obrigatório.</div>", status_code=400)

    try:
        update_values = {'nome_especie': nome_especie.strip(),
                         'pais_origem': pais_origem.strip(),
                         'temperamento': temperamento.strip()}
        where_conditions = {'id': int(gato_id)}
                                                
        rows_affected = GenericRepository.update_table('especie_gatos', update_values, where_conditions)

        if rows_affected == 0:
            print(f"Aviso: Update para gato ID {gato_id} não afetou nenhuma linha.")

        df_updated = GenericRepository.read_table_to_dataframe('especie_gatos', where_conditions={'id': gato_id})
        if df_updated.empty:
            raise HTTPException(status_code=404, detail=f"Gato ID {gato_id} não encontrado após update.")

        gato = df_updated.to_dict('records')[0]
                                                           
        return templates.TemplateResponse("partials/_gato_row.html", {"request": request, "gato": gato, "config": config})
    except HTTPException as http_exc:
         raise http_exc                
    except Exception as e:
        err_str = str(e).lower()
                                                                                 
        if "unique constraint" in err_str or "duplicate entry" in err_str or "constraint failed" in err_str:
             msg = f"Erro: A espécie '{nome_especie}' já existe."
             status = 409           
        else:
            msg = f"Erro ao atualizar: {e}"
            status = 500
        error_html = f"<tr id='gato-row-{gato_id}'><td colspan='4'><div class='alert alert-danger mb-0'>{msg}</div></td></tr>"
        return HTMLResponse(content=error_html, status_code=status)

@router.delete("/{gato_id}", response_class=Response)                               
async def delete_gato(gato_id: int):
    """Processa a exclusão de um gato (via HTMX)."""
    try:
        where_conditions = {'id': int(gato_id)}
                                               
        rows_affected = GenericRepository.delete_from_table('especie_gatos', where_conditions)

        if rows_affected == 0:
            print(f"Aviso: Tentativa de deletar gato ID {gato_id}, mas nenhuma linha foi afetada.")
        return Response(content="", status_code=200)                                                 
    except Exception as e:
        print(f"Erro ao excluir gato ID {gato_id}: {e}")
        return Response(content=f"Erro ao excluir: {e}", status_code=500)

@router.get("/cancelar", response_class=HTMLResponse)
async def cancel_form():
    """Limpa o contêiner do formulário (retorna HTML vazio)."""
    return HTMLResponse("")