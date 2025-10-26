                                                
import json
from pathlib import Path
from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import auth as app_auth
import config                 
from utils.settings import (
    load_theme_settings,
    save_settings as util_save_settings,
    reload_settings,
    get_current_settings,
    AVAILABLE_THEMES as SETTINGS_AVAILABLE_THEMES,
    AVAILABLE_FONTS as SETTINGS_AVAILABLE_FONTS
)

router = APIRouter(
    prefix="/theme",
    tags=["Theme Editor"],
    dependencies=[Depends(app_auth.get_current_user)]
)
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def get_theme_editor_page(request: Request):
    """Exibe a página do editor de temas."""
    settings = get_current_settings()
    if "custom_colors" not in settings or not isinstance(settings["custom_colors"], dict):
         settings["custom_colors"] = {"primary": "#007bff", "secondary": "#6c757d", "success": "#28a745", "info": "#17a2b8", "warning": "#ffc107", "danger": "#dc3545"}

    context = {
        "request": request,
        "settings": settings,
        "available_themes": SETTINGS_AVAILABLE_THEMES,
        "available_fonts": SETTINGS_AVAILABLE_FONTS,
        "message": request.query_params.get("message"),
        "config": config                              
    }
    return templates.TemplateResponse("theme_editor.html", context)

@router.post("/", response_class=RedirectResponse)
async def save_theme_settings_post(
    request: Request,
    theme: str = Form(...),
    font_family: str = Form(...),
    font_size: int = Form(...),
    color_primary: str = Form(...),
    color_secondary: str = Form(...),
    color_success: str = Form(...),
    color_info: str = Form(...),
    color_warning: str = Form(...),
    color_danger: str = Form(...),
    border_width: int = Form(...),
    border_radius: int = Form(...),
    focus_ring: bool = Form(False)
):
    """Salva as configurações."""
    new_settings = {
        "theme": theme, "font_family": font_family, "font_size": font_size,
        "custom_colors": {
            "primary": color_primary, "secondary": color_secondary, "success": color_success,
            "info": color_info, "warning": color_warning, "danger": color_danger
        },
        "border_width": border_width, "border_radius": border_radius, "focus_ring": focus_ring
    }
    if util_save_settings(new_settings):
        reload_settings()
        return RedirectResponse(url="/theme?message=Configurações salvas! Recarregue a página (Ctrl+R) para ver todas as alterações.", status_code=303)
    else:
         raise HTTPException(status_code=500, detail="Erro ao salvar as configurações.")