"""Generate ignored local credentials once, without printing secret values."""
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
env = root / ".env"
if not env.exists():
    password = secrets.token_urlsafe(32)
    env.write_text(f"DATABASE_URL=postgresql+psycopg://securelens:{password}@127.0.0.1:55432/securelens\n"
                   f"LOCAL_POSTGRES_PASSWORD={password}\nJWT_SECRET={secrets.token_urlsafe(48)}\n"
                   'FRONTEND_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]\n'
                   "COOKIE_SECURE=false\nSTORAGE_ROOT=.local/storage\nIMAGE_RETENTION_DAYS=7\n")
    env.chmod(0o600)
    print("Generated backend/.env with private local credentials.")
else:
    print("Existing backend/.env preserved.")
