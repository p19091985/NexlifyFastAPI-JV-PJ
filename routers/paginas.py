                                                  

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import auth as app_auth
from utils.settings import get_current_settings
import config                 

router = APIRouter(
    dependencies=[Depends(app_auth.get_current_user)]
)
templates = Jinja2Templates(directory="templates")

@router.get("/sobre", response_class=HTMLResponse)
async def get_sobre_page(request: Request):
    """Renderiza a página 'Sobre'."""
    settings = get_current_settings()
                                   
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("sobre.html", context)

@router.get("/modos", response_class=HTMLResponse)
async def get_modos_page(request: Request):
    """Renderiza a página 'Alternar Modos'."""
    settings = get_current_settings()
                                   
    context = {"request": request, "settings": settings, "config": config}
    return templates.TemplateResponse("modos.html", context)