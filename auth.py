                                                      

import config
from fastapi import Request, HTTPException, Depends
from starlette.responses import RedirectResponse
from functools import wraps
import logging
import time                 

logger = logging.getLogger(__name__)

SESSION_INACTIVITY_TIMEOUT = 1300

def get_user_session(request: Request) -> dict | None:
    """Obtém as informações do usuário da sessão."""
    user_info = request.session.get("user_info")
                                                                               
    return user_info

def login_user(request: Request, user_data: dict):
    """
    Armazena informações do usuário na sessão.
    Espera um dict com chaves: 'username', 'name', 'access_level'.
    Adiciona timestamp da última atividade.
    """
    session_data = {
        'username': user_data.get('username'),
        'name': user_data.get('name'),
        'access_level': user_data.get('access_level'),
        'last_activity': time.time()                                           
    }
    request.session["user_info"] = session_data
    logger.info(f"Usuário '{session_data.get('username')}' logado na sessão.")

def logout_user(request: Request):
    """Limpa a sessão do usuário."""
    username = request.session.get("user_info", {}).get('username', 'desconhecido')
    request.session.clear()
    logger.info(f"Sessão limpa para usuário '{username}'.")

def get_current_user(request: Request) -> dict:
    """
    Dependência FastAPI: Obtém usuário da sessão ou redireciona para login.
    Se USE_LOGIN=False, cria e loga um usuário 'dev'.
    Implementa timeout de inatividade.
    """
    user_info = get_user_session(request)

    if user_info:
                                                                       
        if config.USE_LOGIN:
            last_activity_time = user_info.get('last_activity', 0)
            current_time = time.time()
            if current_time - last_activity_time > SESSION_INACTIVITY_TIMEOUT:
                logger.warning(f"Sessão expirada por inatividade para '{user_info.get('username', 'N/A')}'.")
                logout_user(request)                          
                                                   
                raise HTTPException(
                    status_code=307,
                    detail="Sessão expirada por inatividade",
                    headers={"Location": "/login"}
                )

        user_info['last_activity'] = time.time()
        request.session['user_info'] = user_info

        return user_info

    if not config.USE_LOGIN:
        logger.warning("Modo DEV: Login desabilitado. Criando usuário 'dev_user'.")
        dev_user_data = {
            'username': 'dev_user',
            'name': 'Usuário de Desenvolvimento',
            'access_level': 'Administrador Global'
                                                                        
        }
        login_user(request, dev_user_data)

        return request.session.get("user_info")

    logger.warning("Usuário não autenticado ou sessão inválida. Redirecionando para /login.")
    raise HTTPException(
        status_code=307,                                          
        detail="Não autenticado",
        headers={"Location": "/login"}                          
    )

def check_permissions(allowed_roles: list):
    """
    Factory de Dependência FastAPI: Verifica se o usuário logado tem o perfil necessário.
    """
    def permission_checker(user: dict = Depends(get_current_user)) -> dict:
                                                            
        if not allowed_roles:
            return user

        user_access_level = user.get('access_level')

        if user_access_level not in allowed_roles:
            logger.warning(f"Acesso negado para '{user.get('username')}' (nível: {user_access_level}). Requerido: {allowed_roles}")
            raise HTTPException(
                status_code=403,            
                detail=f"Acesso negado. Requer perfil: {', '.join(allowed_roles)}"
            )
        return user                                                        

    return permission_checker