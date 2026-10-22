# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Seed derivation with named sub-streams (SPEC §4).

Never use global RNG state. Every random component draws from a stream derived from
(scenario seed, stream name), so adding a new consumer or changing one component does
not reshuffle the others.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _name_key(name: str) -> tuple[int, ...]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # Four 32-bit words of the digest form the spawn key.
    return tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))


def seed_sequence(seed: int, *names: str) -> np.random.SeedSequence:
    """SeedSequence for the stream `names` (hierarchical, e.g. ("noise", "target"))."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    key: tuple[int, ...] = ()
    for n in names:
        key = key + _name_key(n)
    return np.random.SeedSequence(entropy=seed, spawn_key=key)


def generator(seed: int, *names: str) -> np.random.Generator:
    """A numpy Generator (PCG64) for the named stream."""
    return np.random.Generator(np.random.PCG64(seed_sequence(seed, *names)))


def derive_int(seed: int, *names: str) -> int:
    """A derived non-negative 63-bit integer seed for the named stream."""
    word = seed_sequence(seed, *names).generate_state(2, dtype=np.uint32)
    return (int(word[0]) << 31) ^ int(word[1])


def jax_key(seed: int, *names: str) -> Any:
    """A JAX PRNG key for the named stream."""
    import jax

    return jax.random.PRNGKey(derive_int(seed, *names) & 0x7FFFFFFF)
