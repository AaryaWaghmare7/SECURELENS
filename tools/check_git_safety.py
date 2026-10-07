"""Read-only checks for common secrets, private data and large Git candidates.

This scans the current checkout/index, not historical commits, and is not a
replacement for a dedicated secret scanner or review before publishing.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b"),
    re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}\b"),
)
DATABASE_CREDENTIALS = re.compile(r"postgres(?:ql)?(?:\+psycopg)?://[^\s:/@]+:([^\s/@]+)@")


def findings(path, text):
    issues = []
    parts = Path(path).parts
    if (any(part in {".local", "node_modules", "media", "data", ".venv", "venv"} for part in parts) or
            any(part.startswith(".env") and part != ".env.example" for part in parts) or
            Path(path).suffix.lower() in {".sqlite3", ".db", ".pth", ".keras", ".h5"}):
        issues.append("private/runtime file is a Git candidate")
    for number, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            issues.append(f"line {number}: possible credential (value not printed)")
        for match in DATABASE_CREDENTIALS.finditer(line):
            password = match.group(1)
            if not any(marker in password.lower() for marker in ("replace", "example", "placeholder", "${", "{password}")):
                issues.append(f"line {number}: possible database password (value not printed)")
    return issues


def main():
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    candidates = sorted(set(output.decode().split("\0")) - {""})
    failures = []
    checked = 0
    for name in candidates:
        path = ROOT / name
        if not path.is_file():
            continue
        checked += 1
        if path.stat().st_size > MAX_BYTES:
            failures.append((name, "file exceeds 10 MB; review before publishing"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            content = ""
        failures.extend((name, issue) for issue in findings(name, content))
    for name, issue in failures:
        print(f"REVIEW {name}: {issue}")
    print(f"Checked {checked} Git candidate files; {len(failures)} review finding(s).")
    print("Read-only check. Git history and uncommon secret formats still require review.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
