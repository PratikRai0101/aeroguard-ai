#!/usr/bin/env python3
"""
AeroGuard AI - One-run setup & launch script.

Works on Windows, macOS, and Linux.
1. Ensures a Python virtual environment exists.
2. Installs backend dependencies.
3. Checks for Ollama, offers to install it if missing.
4. Pulls the configured SLM (default: qwen3.5:2b).
5. Starts Ollama if not running.
6. Starts the FastAPI backend.

Usage:
    python setup_and_run.py
"""

import os
import sys
import time
import json
import platform
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# Make sure status messages appear immediately when the script is redirected to a log file.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
VENV_DIR = PROJECT_ROOT / "venv"
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def parse_args():
    """Parse simple command-line flags."""
    skip = False
    for arg in sys.argv[1:]:
        if arg in ("--skip-model-pull", "-s"):
            skip = True
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python setup_and_run.py [--skip-model-pull]")
            sys.exit(1)
    return skip


def run(cmd, cwd=None, check=True):
    """Run a shell command and print it."""
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def python_bin(name="python"):
    """Return the path to the Python binary inside the venv."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / f"{name}.exe")
    return str(VENV_DIR / "bin" / name)


def ensure_venv():
    """Create virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("Virtual environment already exists.")


def install_deps():
    """Install backend dependencies."""
    pip = python_bin("pip")
    run([pip, "install", "--upgrade", "pip"])
    run([pip, "install", "-r", str(BACKEND_DIR / "requirements.txt")])


def find_ollama():
    """Find the Ollama executable path."""
    candidates = []
    system = platform.system()

    if system == "Windows":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            Path(localappdata) / "Programs" / "Ollama" / "ollama.exe",
            Path("C:") / "Program Files" / "Ollama" / "ollama.exe",
        ]
    elif system == "Darwin":
        # The app-bundle executable is the GUI launcher.  Use Ollama's CLI
        # binary instead so `ollama serve` starts the local API server.
        candidates = [
            Path("/usr/local/bin/ollama"),
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        ]
    else:
        candidates = [
            Path.home() / ".local" / "bin" / "ollama",
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fall back to PATH lookup
    for path in os.environ.get("PATH", "").split(os.pathsep):
        exe = Path(path) / ("ollama.exe" if system == "Windows" else "ollama")
        if exe.exists():
            return str(exe)

    return None


def download_file(url, dest):
    """Download a file with a simple progress indicator."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def install_ollama():
    """Attempt to install Ollama automatically."""
    system = platform.system()

    if system == "Windows":
        installer = PROJECT_ROOT / "OllamaSetup.exe"
        if not installer.exists():
            download_file("https://ollama.com/download/OllamaSetup.exe", str(installer))
        print("\nInstalling Ollama...")
        run([str(installer), "/S"])
        print("Ollama installed. You may need to restart your terminal.")
        return True

    elif system == "Darwin":
        print("\nInstalling Ollama on macOS...")
        run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        return True

    else:
        print("\nInstalling Ollama on Linux...")
        run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        return True


def _ollama_api_get(path, timeout=5):
    """Make a GET request to the Ollama API using only stdlib."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}{path}", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[Ollama API] {path} failed: {e}")
        return None


def _ollama_api_post(path, payload, timeout=600):
    """Make a POST request to the Ollama API using only stdlib."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[Ollama API] {path} failed: {e}")
        return None


def ensure_ollama():
    """Ensure Ollama is installed and running."""
    ollama_path = find_ollama()

    if not ollama_path:
        print("\nOllama is not installed.")
        answer = input("Download and install Ollama now? (y/n): ").strip().lower()
        if answer == "y":
            install_ollama()
            ollama_path = find_ollama()
            if not ollama_path:
                print("Ollama installation not found. Please install it manually from https://ollama.com")
                sys.exit(1)
        else:
            print("Please install Ollama from https://ollama.com and rerun.")
            sys.exit(1)

    print(f"Ollama found: {ollama_path}")

    # Check if Ollama server is running
    if _ollama_api_get("/api/tags", timeout=3):
        print("Ollama server is already running.")
        return ollama_path

    print("Starting Ollama server...")
    if platform.system() == "Windows":
        subprocess.Popen([ollama_path, "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for server to be ready
    for _ in range(30):
        if _ollama_api_get("/api/tags", timeout=2):
            print("Ollama server is ready.")
            return ollama_path
        time.sleep(1)

    print("Ollama server did not start in time. Please start it manually.")
    return ollama_path


def pull_model(ollama_path, skip=False):
    """Pull the configured SLM if not already present."""
    if skip:
        print(f"Skipping model pull for {MODEL} (use the backend endpoint to pull on demand).")
        return

    tags = _ollama_api_get("/api/tags", timeout=5)
    if tags:
        models = [m.get("name", "") for m in tags.get("models", [])]
        if any(MODEL in name or name in MODEL for name in models):
            print(f"Model {MODEL} is already available.")
            return
    else:
        print("Could not check models.")

    print(f"Pulling model {MODEL}. This may take a few minutes...")
    print("Tip: to skip this step, run with --skip-model-pull")
    run([ollama_path, "pull", MODEL])


def start_backend():
    """Start the FastAPI backend."""
    python = python_bin("python")
    print("\nStarting AeroGuard AI backend...")
    print("API docs: http://localhost:8000/docs")
    print("Keep this terminal open. Start the mobile app from a second terminal:")
    print("  cd mobile && npx expo start")
    run([python, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])


def main():
    skip_model_pull = parse_args()

    print("=" * 60)
    print("AeroGuard AI - Setup & Launch")
    print("=" * 60)

    ensure_venv()
    install_deps()
    ollama_path = ensure_ollama()
    pull_model(ollama_path, skip=skip_model_pull)
    start_backend()


if __name__ == "__main__":
    main()
