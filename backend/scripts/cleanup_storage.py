"""Run periodically to enforce image retention even when users do not visit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.database.session import make_database
from app.services.records import cleanup_expired
from app.services.storage import LocalStorage

settings = Settings()
engine, factory = make_database(settings.database_url)
with factory() as session:
    cleanup_expired(session, LocalStorage(settings.storage_path))
engine.dispose()
print("Expired image previews removed; analysis metrics and reports retained.")
