"""Index of produced figures and animations (`figures.json`, SPEC §7.1.1).

Reports are built from manifests, never by globbing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

FigureKind = Literal["figure", "animation", "table"]


@dataclass
class FigureEntry:
    name: str
    kind: FigureKind
    paths: list[str]
    """Relative to the manifest's directory."""
    title: str
    caption: str
    context: dict[str, Any] = field(default_factory=dict[str, Any])
    """scenario / strategy / seed (and any parameters) the figure depicts."""


class FigureManifest:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.entries: list[FigureEntry] = []

    @property
    def path(self) -> Path:
        return self.root / "figures.json"

    def add(
        self,
        *,
        name: str,
        kind: FigureKind,
        paths: list[Path],
        title: str,
        caption: str,
        context: dict[str, Any],
    ) -> FigureEntry:
        rel = [str(Path(p).resolve().relative_to(self.root.resolve())) for p in paths]
        entry = FigureEntry(name, kind, rel, title, caption, dict(context))
        self.entries = [e for e in self.entries if e.name != name] + [entry]
        return entry

    def get(self, name: str) -> FigureEntry | None:
        return next((e for e in self.entries if e.name == name), None)

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {"figures": [asdict(e) for e in self.entries]}
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, root: str | Path) -> FigureManifest:
        m = cls(root)
        if m.path.exists():
            data = json.loads(m.path.read_text(encoding="utf-8"))
            m.entries = [FigureEntry(**e) for e in data.get("figures", [])]
        return m

    def __enter__(self) -> FigureManifest:
        return self

    def __exit__(self, *exc: object) -> None:
        self.save()
