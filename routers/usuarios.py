                                                             

import pandas as pd
from fastapi import APIRouter, Request, Depends, Form, HTTPException
                                   
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from persistencia.repository import GenericRepository
from persistencia.auth import hash_password
import auth as app_auth
import config
from utils.settings import get_current_settings

ACCESS_ROLES = ['Administrador Global', 'Gerente de TI']

router = APIRouter(
    prefix="/usuarios",
    tags=["Usuários"],
    dependencies=[Depends(app_auth.check_permissions(ACCESS_ROLES))]
)
templates = Jinja2Templates(directory="templates")

PERFIS_DE_ACESSO = [
    'Administrador Global', 'Diretor de Operações', 'Gerente de TI',
    'Supervisor de Produção', 'Operador de Linha', 'Analista de Dados', 'Auditor Externo'
]

@router.get("/", response_class=HTMLResponse)
async def get_usuarios_page(request: Request):
    """Renderiza a página principal de Gestão de Usuários."""
    settings = get_current_settings()
    users_list = []
    db_disabled = not config.DATABASE_ENABLED
    error_message = None

    if not db_disabled:
        try:
            df = GenericRepository.read_table_to_dataframe("usuarios")
            if 'senha_criptografada' in df.columns:
                df = df.drop(columns=['senha_criptografada'])
            users_list = df.to_dict('records')
        except Exception as e:
            error_message = f"Erro ao carregar usuários: {e}"
            print(error_message)

    context = {
        "request": request,
        "settings": settings,
        "usuarios": users_list,
        "db_disabled": db_disabled,
        "error_message": error_message,
        "config": config                              
    }
    return templates.TemplateResponse("usuarios.html", context)

@router.get("/novo", response_class=HTMLResponse)
async def get_user_form(request: Request):
    """Retorna o formulário HTMX para adicionar um novo usuário."""

    return templates.TemplateResponse(
        "partials/_usuario_form.html",
        {"request": request, "perfis": PERFIS_DE_ACESSO, "config": config}
    )

@router.get("/{login_usuario}/editar", response_class=HTMLResponse)
async def get_user_edit_form(request: Request, login_usuario: str):
    """Retorna o formulário HTMX preenchido para editar um usuário."""
    try:
                                                                
        df_all = GenericRepository.read_table_to_dataframe('usuarios')

        df_user = df_all[df_all['login_usuario'] == login_usuario]

        if df_user.empty:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")

        user = df_user.to_dict('records')[0]

        return templates.TemplateResponse(
            "partials/_usuario_form.html",
            {"request": request, "usuario": user, "perfis": PERFIS_DE_ACESSO, "config": config}
        )
    except Exception as e:
                                                   
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao carregar usuário: {e}</div>", status_code=500)

@router.post("/", response_class=HTMLResponse)
async def create_user(
        request: Request,
        login_usuario: str = Form(...),
        nome_completo: str = Form(...),
        tipo_acesso: str = Form(...),
        password: str = Form(...)
):
    """Processa a criação de um novo usuário (via HTMX)."""
    if not login_usuario.strip() or not nome_completo.strip():
                                               
        return HTMLResponse("<div class='alert alert-danger'>Login e Nome são obrigatórios.</div>",
                            status_code=400)
    if not password:
                                               
        return HTMLResponse(
            "<div class='alert alert-danger'>Senha é obrigatória para novos usuários.</div>",
            status_code=400)

    try:
        hashed_pw = hash_password(password)
        df = pd.DataFrame([{
            'login_usuario': login_usuario.strip(),
            'senha_criptografada': hashed_pw,
            'nome_completo': nome_completo.strip(),
            'tipo_acesso': tipo_acesso
        }])

        GenericRepository.write_dataframe_to_table(df, "usuarios")

        return Response(content="", headers={"HX-Refresh": "true"})

    except Exception as e:
                                              
        err_str = str(e).lower()
        if "unique constraint" in err_str or "duplicate entry" in err_str or "constraint failed" in err_str:
            msg = f"Erro: O login '{login_usuario}' já existe."
            status = 409            
        else:
            msg = f"Erro ao salvar: {e}"
            status = 500

        return HTMLResponse(f"<div class='alert alert-danger'>{msg}</div>", status_code=status)

@router.put("/{login_usuario}", response_class=HTMLResponse)
async def update_user(
        request: Request,
        login_usuario: str,
        nome_completo: str = Form(...),
        tipo_acesso: str = Form(...),
        password: str = Form(None)
):
    """Processa a atualização de um usuário existente (via HTMX)."""
    if not nome_completo.strip():
                                               
        return HTMLResponse("<div class='alert alert-danger'>Nome Completo é obrigatório.</div>",
                            status_code=400)

    try:
        update_values = {
            'nome_completo': nome_completo.strip(),
            'tipo_acesso': tipo_acesso
        }
        if password and password.strip():
            hashed_pw = hash_password(password)
            update_values['senha_criptografada'] = hashed_pw

        where_conditions = {'login_usuario': login_usuario}

        rows_affected = GenericRepository.update_table("usuarios", update_values, where_conditions)

        if rows_affected == 0:
            print(f"Aviso: Update para usuário '{login_usuario}' não afetou nenhuma linha.")

        return Response(content="", headers={"HX-Refresh": "true"})

    except Exception as e:
        print(f"Erro ao atualizar usuário '{login_usuario}': {e}")
                                               
        return HTMLResponse(f"<div class='alert alert-danger'>Erro ao atualizar: {e}</div>",
                            status_code=500)

@router.delete("/{login_usuario}", response_class=HTMLResponse)
async def delete_user(login_usuario: str):
    """Processa a exclusão de um usuário (via HTMX)."""
    try:
        where_conditions = {'login_usuario': login_usuario}
        rows_affected = GenericRepository.delete_from_table("usuarios", where_conditions)

        if rows_affected == 0:
            print(f"Aviso: Tentativa de deletar usuário '{login_usuario}', mas nenhuma linha foi afetada.")

        return Response(content="", headers={"HX-Refresh": "true"})

    except Exception as e:
        print(f"Erro ao excluir usuário '{login_usuario}': {e}")
        return HTMLResponse(f"Erro ao excluir: {e}", status_code=500)

@router.get("/cancelar", response_class=HTMLResponse)
async def cancel_form():
    """Limpa o contêiner do formulário (retorna HTML vazio)."""
    return HTMLResponse("")