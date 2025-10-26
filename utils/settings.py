                                                           
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
SETTINGS_FILE = Path("settings.json")

AVAILABLE_THEMES = [
    "lumen", "darkly", "cerulean", "cosmo", "flatly", "journal", "litera", "lux",
    "materia", "minty", "pulse", "sandstone", "simplex", "sketchy", "slate",
    "solar", "spacelab", "superhero", "united", "vapor", "yeti", "zephyr"
]
AVAILABLE_FONTS = [
    "Segoe UI", "Helvetica", "Arial", "Verdana", "Roboto", "Noto Sans"
]

DEFAULT_SETTINGS = {
    "theme": "lumen",                               
    "font_family": 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", "Liberation Sans", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
    "font_size": 16,                                                                      
    "custom_colors": {
        "primary": "#0d6efd",
        "secondary": "#6c757d",
        "success": "#198754",
        "info": "#0dcaf0",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "light": "#f8f9fa",
        "dark": "#212529"
    },
    "border_width": 1,         
    "border_radius": 6,                                                              
    "focus_ring": True
}
                                                                         
DARK_THEMES = ["darkly", "slate", "solar", "superhero", "vapor", "cyborg"]

_settings_cache = None

def hex_to_rgb_tuple(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        try:
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return (0, 0, 0)                                 
    elif len(hex_color) == 3:                        
        try:
            return tuple(int(hex_color[i:i + 1] * 2, 16) for i in (0, 1, 2))
        except ValueError:
            return (0, 0, 0)
    return (0, 0, 0)

def load_theme_settings(force_reload: bool = False) -> dict:
    """Carrega as configurações do settings.json, com cache."""
    global _settings_cache
    if _settings_cache is not None and not force_reload:
        return _settings_cache

    if not SETTINGS_FILE.exists():
        logger.warning(f"Arquivo '{SETTINGS_FILE}' não encontrado. Usando configurações padrão.")
        _settings_cache = DEFAULT_SETTINGS.copy()
                                                
        if "custom_colors_rgb" not in _settings_cache:
            _settings_cache["custom_colors_rgb"] = {k: hex_to_rgb_tuple(v) for k, v in
                                                    _settings_cache.get("custom_colors", {}).items()}
        return _settings_cache
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
                                                                        
            for key, default_value in DEFAULT_SETTINGS.items():
                if key not in settings:
                    settings[key] = default_value
                                                                           
                elif key == "custom_colors" and isinstance(default_value, dict):
                    if not isinstance(settings[key], dict):
                        settings[key] = default_value                             
                    else:
                        for color_key, color_default in default_value.items():
                            if color_key not in settings[key]:
                                settings[key][color_key] = color_default

            settings["custom_colors_rgb"] = {k: hex_to_rgb_tuple(v) for k, v in
                                             settings.get("custom_colors", {}).items()}

            _settings_cache = settings
            logger.info(f"Configurações de tema carregadas de '{SETTINGS_FILE}'.")
            return _settings_cache
    except (json.JSONDecodeError, IOError, Exception) as e:
        logger.error(f"Erro ao carregar ou parsear '{SETTINGS_FILE}': {e}. Usando configurações padrão.")
        _settings_cache = DEFAULT_SETTINGS.copy()
                                                
        if "custom_colors_rgb" not in _settings_cache:
            _settings_cache["custom_colors_rgb"] = {k: hex_to_rgb_tuple(v) for k, v in
                                                    _settings_cache.get("custom_colors", {}).items()}
        return _settings_cache

def save_settings(settings: dict):
    """Salva as configurações no arquivo JSON."""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                                                                                
            settings_to_save = settings.copy()
            settings_to_save.pop("custom_colors_rgb", None)
            json.dump(settings_to_save, f, indent=4)
        logger.info(f"Configurações salvas em '{SETTINGS_FILE}'.")
        return True
    except IOError as e:
        logger.error(f"Erro ao salvar settings.json: {e}")
        return False
    except Exception as e:
        logger.error(f"Erro inesperado ao salvar settings.json: {e}")
        return False

def get_current_settings() -> dict:
    """Retorna as configurações cacheadas ou carrega se ainda não foram."""
    if _settings_cache is None:
        return load_theme_settings()
    return _settings_cache

def reload_settings() -> dict:
    """Força o recarregamento das configurações do arquivo."""
    logger.info(f"Recarregando configurações de '{SETTINGS_FILE}'.")
    return load_theme_settings(force_reload=True)

def get_theme_mode(settings: dict = None) -> str:
    """Determina se o tema é 'light' ou 'dark'."""
    if settings is None:
        settings = get_current_settings()
    theme_name = settings.get("theme", "lumen").lower()
    return "dark" if theme_name in DARK_THEMES else "light"

def px_to_rem(px_value: int, base_font_size: int = 16) -> str:
    """Converte um valor em pixels para rem string (ex: '0.375rem')."""
    try:
        rem_value = round(float(px_value) / base_font_size, 3)
        return f"{rem_value}rem"
    except (ValueError, TypeError):
                                                                          
        return "0.375rem"