"""The information boundary (SPEC §3.2) is enforced by import-linter contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "jt_core": "jt-core",
    "tycho": "tycho",
    "jan": "jan",
    "rudolf": "rudolf",
    "jt_bench": "jt-bench",
}


def _lint(config: Path, pythonpath: list[Path]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports; "
            f"sys.exit(lint_imports(config_filename={str(config)!r}, no_cache=True))",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=config.parent,
    )


def test_contracts_hold() -> None:
    srcs = [ROOT / "packages" / d / "src" for d in PACKAGES.values()]
    result = _lint(ROOT / ".importlinter", srcs)
    assert result.returncode == 0, result.stdout + result.stderr


def test_deliberate_violation_is_caught(tmp_path: Path) -> None:
    """A deliberate `import tycho` inside `jan` must break the build."""
    srcs: list[Path] = []
    for mod, d in PACKAGES.items():
        src = tmp_path / d / "src"
        shutil.copytree(ROOT / "packages" / d / "src" / mod, src / mod)
        srcs.append(src)
    (tmp_path / "jan" / "src" / "jan" / "_leak.py").write_text("import tycho  # noqa\n")
    shutil.copy(ROOT / ".importlinter", tmp_path / ".importlinter")
    result = _lint(tmp_path / ".importlinter", srcs)
    assert result.returncode != 0
    assert "jan -> tycho" in result.stdout or "BROKEN" in result.stdout
