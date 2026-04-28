"""
start.py — One-click setup & launch script.

Run:  python start.py

This script:
  1. Checks Python version
  2. Creates a .env file from .env.example (if missing)
  3. Installs requirements
  4. Downloads the spaCy model
  5. Downloads NLTK data
  6. Launches the FastAPI server
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
ENV_FILE = BACKEND_DIR / ".env"
ENV_EXAMPLE = BACKEND_DIR / ".env.example"
REQUIREMENTS = BACKEND_DIR / "requirements.txt"


def run(cmd: list[str], **kwargs) -> int:
    print(f"\n▶  {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def check_python() -> None:
    major, minor = sys.version_info[:2]
    print(f"🐍 Python {major}.{minor} detected.")
    if major < 3 or (major == 3 and minor < 9):
        print("❌ Python 3.9+ is required.")
        sys.exit(1)


def ensure_env() -> None:
    if not ENV_FILE.exists() and ENV_EXAMPLE.exists():
        import shutil
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        print("📄 Created .env from .env.example — edit it to add your OpenAI key (optional).")
    else:
        print("📄 .env file found.")


def install_deps() -> None:
    print("\n📦 Installing Python dependencies …")
    code = run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS), "--quiet"])
    if code != 0:
        print("⚠  pip install had warnings/errors (may be okay). Continuing …")


def download_spacy_model() -> None:
    print("\n🔤 Downloading spaCy model (en_core_web_sm) …")
    try:
        import spacy  # noqa: F401
        try:
            import en_core_web_sm  # noqa: F401
            print("✅ spaCy model already installed.")
            return
        except ImportError:
            pass
    except ImportError:
        pass
    run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def download_nltk_data() -> None:
    print("\n📚 Downloading NLTK data …")
    try:
        import nltk  # noqa: F401
        for resource in ["punkt", "averaged_perceptron_tagger", "wordnet", "stopwords"]:
            nltk.download(resource, quiet=True)
        print("✅ NLTK data ready.")
    except Exception as exc:
        print(f"⚠  NLTK download warning: {exc}")


def launch_server() -> None:
    print("\n🚀 Starting AMDOX AI Chatbot server …")
    print("=" * 55)
    print("  API:       http://localhost:8000")
    print("  Docs:      http://localhost:8000/docs")
    print("  Analytics: http://localhost:8000/api/analytics")
    print("=" * 55)
    print("  Press Ctrl+C to stop.\n")
    os.chdir(BACKEND_DIR)
    run([sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"])


if __name__ == "__main__":
    check_python()
    ensure_env()
    install_deps()
    download_spacy_model()
    download_nltk_data()
    launch_server()
