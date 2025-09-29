# app/deps.py
import os
from fastapi import Header, HTTPException, status

API_KEY_ENV = "API_KEY"

def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        # API_KEY yoksa, doğrulama devre dışı (dev kolaylığı)
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
