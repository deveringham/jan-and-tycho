"""Jan: the theorist. Inverse-problem strategies that see only public data.

Jan must never import `tycho` or `rudolf` (enforced by import-linter, SPEC §3.2).
"""

from jan.base import get_strategy, list_strategies, register

__all__ = ["get_strategy", "list_strategies", "register"]
