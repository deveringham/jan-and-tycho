# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Shared, truth-agnostic utilities for Jan and Tycho.

Importing this package enables 64-bit floats in JAX globally (SPEC §4): every
component must compute in float64.
"""

import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

# Persistent XLA compilation cache: the conic solver takes ~1 s to compile per shape, and
# tests / sweep workers run in separate processes. Set JT_JAX_CACHE="" to disable.
_cache = os.environ.get("JT_JAX_CACHE", str(Path.home() / ".cache" / "jan-and-tycho" / "jax"))
if _cache:
    jax.config.update("jax_compilation_cache_dir", _cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)

__version__ = "0.1.0"
