"""Start the local React/FastAPI/PostgreSQL workspace; never deploys."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]


def available(port):
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", port)) == 0


def runtime():
    node = shutil.which("node")
    npm = shutil.which("npm")
    pnpm = shutil.which("pnpm")
    if not node:
        # Optional Codex-bundled runtime; ordinary Node installations take priority.
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
        binary = bundled / "node/bin/node"
        manager = bundled / "bin/fallback/pnpm"
        if binary.exists() and manager.exists():
            node, pnpm = str(binary), str(manager)
    if not node or not (npm or pnpm):
        raise RuntimeError("Install Node.js 22.12+ (24 recommended) and npm or pnpm first.")
    os.environ["PATH"] = str(Path(node).parent) + os.pathsep + os.environ.get("PATH", "")
    return node, [pnpm or npm]


def run(command, directory):
    subprocess.run(command, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true", help="Install local dependencies if needed.")
    parser.add_argument("--api-port", type=int, default=8000, help="Use another API port if Django is already running.")
    options = parser.parse_args()
    node, manager = runtime()
    python = next((path for path in (ROOT / "backend/.venv/bin/python", ROOT / ".venv/bin/python",
                                    ROOT / "backend/.venv/Scripts/python.exe", ROOT / ".venv/Scripts/python.exe") if path.exists()), None)
    if not python:
        if not options.setup:
            raise RuntimeError("First run: python3 tools/dev.py --setup")
        venv.create(ROOT / "backend/.venv", with_pip=True)
        python = ROOT / ("backend/.venv/Scripts/python.exe" if os.name == "nt" else "backend/.venv/bin/python")
    if options.setup:
        run([str(python), "-m", "pip", "install", "-r", "requirements-dev.txt"], ROOT / "backend")
    run([str(python), "scripts/setup_local.py"], ROOT / "backend")
    for directory in (ROOT / "frontend", ROOT / "tools/local-postgres"):
        if not (directory / "node_modules").exists():
            if not options.setup:
                raise RuntimeError("Dependencies missing. Run python3 tools/dev.py --setup first.")
            run(manager + ["install"], directory)
    occupied = [port for port in (options.api_port, 5173) if available(port)]
    if occupied:
        ports = ", ".join(map(str, occupied))
        raise RuntimeError(f"Port(s) {ports} already in use. If SecureLens is running, open http://127.0.0.1:5173. "
                           "Otherwise stop the old launcher with Ctrl+C in its terminal before restarting. "
                           "--api-port only changes the backend port; it does not free frontend port 5173.")
    os.environ["VITE_PROXY_TARGET"] = f"http://127.0.0.1:{options.api_port}"
    children = []

    def launch(command, directory):
        process = subprocess.Popen(command, cwd=directory, start_new_session=os.name != "nt")
        children.append(process)
        return process

    try:
        from urllib.parse import urlparse
        # Only start the bundled development database for the generated local URL.
        # Read only the database location here; never print credentials.
        values = dict(line.split("=", 1) for line in (ROOT / "backend/.env").read_text().splitlines() if "=" in line and not line.startswith("#"))
        database = urlparse(values["DATABASE_URL"].replace("postgresql+psycopg", "postgresql"))
        if database.hostname in ("127.0.0.1", "localhost") and database.port == 55432 and not available(55432):
            postgres = launch([node, "start.mjs"], ROOT / "tools/local-postgres")
            deadline = time.monotonic() + 45
            while not available(55432):
                if postgres.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Local PostgreSQL did not start. Check the database log above.")
                time.sleep(.2)
        run([str(python), "-m", "alembic", "upgrade", "head"], ROOT / "backend")
        launch([str(python), "-m", "uvicorn", "app.main:app", "--reload", "--reload-dir", str(ROOT / "backend/app"),
                "--reload-dir", str(ROOT / "src"), "--host", "127.0.0.1", "--port", str(options.api_port)], ROOT / "backend")
        launch(manager + ["run", "dev"], ROOT / "frontend")
        print(f"\nSecureLens: http://127.0.0.1:5173\nAPI docs: http://127.0.0.1:{options.api_port}/docs\nCtrl+C stops services started by this command.\n", flush=True)
        while all(process.poll() is None for process in children):
            time.sleep(.5)
        raise RuntimeError("A development service stopped. Check the logs above.")
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(children):
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"SecureLens setup: {error}", file=sys.stderr)
        sys.exit(1)
