from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_markdown_links_and_domains_are_clean() -> None:
    """No-network doc hygiene gate: relative links should resolve."""
    project_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(project_root / "scripts" / "check_markdown_links.py")]
    completed = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, (
        "Markdown link/domain check failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
    )

