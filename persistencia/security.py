                          
import os
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken

KEY_PATH = Path(__file__).parent.parent / "secret.key"

def generate_and_save_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as key_file:
        key_file.write(key)
    return key

def load_key():
    if not KEY_PATH.exists():
        return generate_and_save_key()
    with open(KEY_PATH, "rb") as key_file:
        return key_file.read()

def encrypt_message(message: str, key: bytes) -> str:
    if not message:
        return ""
    f = Fernet(key)
    return f.encrypt(message.encode('utf-8')).decode('utf-8')

def decrypt_message(encrypted_message: str, key: bytes) -> str:
    if not encrypted_message:
        return ""
    f = Fernet(key)
    try:
        return f.decrypt(encrypted_message.encode('utf-8')).decode('utf-8')
    except (InvalidToken, TypeError, AttributeError):
        return encrypted_message