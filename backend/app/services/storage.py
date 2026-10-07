"""Private storage interface; replace LocalStorage with a cloud adapter later."""
from pathlib import Path
from typing import Protocol
import os


class Storage(Protocol):
    def write(self, key: str, content: bytes) -> None: ...
    def read(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, key):
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Invalid storage reference")
        return path

    def write(self, key, content):
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, key):
        return self.path(key).read_bytes()

    def delete(self, key):
        self.path(key).unlink(missing_ok=True)
