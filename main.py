                                                             

import sys
import uvicorn
import config                                                            
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os
import logging

from persistencia.database import DatabaseManager

try:
    from persistencia.logger import setup_loggers
    setup_loggers()
except ImportError:
    print("AVISO: Módulo persistencia.logger não encontrado, usando logger padrão.")
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

logger = logging.getLogger(__name__)

from routers import paginas, gatos, usuarios, iris, covertype, theme
import auth as app_auth
from persistencia import auth as persistencia_auth                              
from utils.settings import get_current_settings

load_dotenv()
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
if not SESSION_SECRET_KEY or SESSION_SECRET_KEY == "SUA_CHAVE_SECRETA_MUITO_FORTE_VAI_AQUI":
    logger.critical("SESSION_SECRET_KEY não definida ou está insegura no arquivo .env!")
    SESSION_SECRET_KEY = "fallback-insecure-secret-key-please-set-env-properly"

def validar_configuracoes():
    """Verifica se a combinação de flags no config.py é válida."""
    logger.info("Validando configurações de config_settings.ini...")
    if config.USE_LOGIN and not config.DATABASE_ENABLED:
        logger.critical("❌ ERRO DE CONFIGURAÇÃO INVÁLIDA ❌")
        logger.critical("Problema: USE_LOGIN=True e DATABASE_ENABLED=False.")
        logger.critical("Motivo: O sistema de login requer o banco de dados.")
        logger.critical("Solução: Defina DATABASE_ENABLED=True ou USE_LOGIN=False.")
        sys.exit(1)

    if config.INITIALIZE_DATABASE_ON_STARTUP and not config.DATABASE_ENABLED:
        logger.critical("❌ ERRO DE CONFIGURAÇÃO INVÁLIDA ❌")
        logger.critical("Problema: INITIALIZE_DATABASE_ON_STARTUP=True e DATABASE_ENABLED=False.")
        logger.critical("Motivo: Não é possível inicializar um banco de dados desativado.")
        logger.critical("Solução: Defina DATABASE_ENABLED=True ou INITIALIZE_DATABASE_ON_STARTUP=False.")
        sys.exit(1)
    logger.info("Configurações validadas com sucesso.")

app = FastAPI(title="Nexlify FastAPI", description="Aplicação migrada do Streamlit.")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    https_only=False,                                       
    max_age=None                                            
)

templates = Jinja2Templates(directory="templates")
                                     
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def on_startup():
    validar_configuracoes()
    if config.DATABASE_ENABLED:
        try:
            logger.info("Tentando inicializar o banco de dados (se necessário)...")
            DatabaseManager.initialize_database()
            logger.info("Verificação de inicialização do banco de dados concluída.")
        except Exception as e:
            logger.critical(f"Erro fatal ao inicializar o banco de dados: {e}")
            sys.exit(1)
    else:
        logger.warning("DATABASE_ENABLED=False. Pulando inicialização do banco de dados.")
    logger.info("Aplicação iniciada.")

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Aplicação encerrada.")

@app.get("/login", response_class=HTMLResponse, tags=["Autenticação"])
async def get_login_page(request: Request):
    user = app_auth.get_user_session(request)
    if user and config.USE_LOGIN:
        return RedirectResponse(url="/", status_code=303)
    settings = get_current_settings()
                                                                      
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("login.html", context)

@app.post("/login", response_class=HTMLResponse, tags=["Autenticação"])
async def post_login(request: Request, username: str = Form(...), password: str = Form(...)):
    if not config.DATABASE_ENABLED:
        logger.error("Tentativa de login com banco desabilitado.")
        settings = get_current_settings()
                                     
        context = {"request": request, "settings": settings, "error": "Login desabilitado: Banco de dados não está ativo.", "config": config}
        return templates.TemplateResponse("login.html", context, status_code=400)

    user_data = persistencia_auth.verify_user_credentials(username, password)

    if user_data == "connection_error":
        logger.error("Falha na conexão com o banco durante o login.")
        settings = get_current_settings()
                                      
        context = {"request": request, "settings": settings, "error": "Falha na conexão com o banco de dados.", "config": config}
        return templates.TemplateResponse("login.html", context, status_code=500)
    elif user_data:
        app_auth.login_user(request, user_data)
        logger.info(f"Login bem-sucedido para: {username}")
        return RedirectResponse(url="/", status_code=303)
    else:
        logger.warning(f"Tentativa de login falhou para: {username}")
        settings = get_current_settings()
                                      
        context = {"request": request, "settings": settings, "error": "Usuário ou senha inválidos.", "config": config}
        return templates.TemplateResponse("login.html", context, status_code=401)

@app.get("/logout", tags=["Autenticação"])
async def logout(request: Request):
    app_auth.logout_user(request)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/", dependencies=[Depends(app_auth.get_current_user)], tags=["Navegação"])
async def get_home(request: Request):
    logger.debug(f"Acesso à rota raiz '/' por usuário '{request.session.get('user_info', {}).get('username', 'N/A')}', redirecionando para /sobre.")
    return RedirectResponse(url="/sobre", status_code=303)

app.include_router(paginas.router)
app.include_router(gatos.router)
app.include_router(usuarios.router)
app.include_router(iris.router)
app.include_router(covertype.router)
app.include_router(theme.router)

if __name__ == "__main__":
    logger.info("Iniciando servidor Nexlify FastAPI diretamente...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL_STR.lower()
    )