# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Physical constants in the project's unit system: AU, days, radians (SPEC §4)."""

import math

K_GAUSS: float = 0.01720209895
"""Gaussian gravitational constant k, in AU^(3/2) / day."""

GM_SUN: float = K_GAUSS**2
"""Heliocentric gravitational parameter k², in AU³ / day²."""

AU: float = 1.0
"""Astronomical unit (the unit of length)."""

AU_KM: float = 149_597_870.7
"""Astronomical unit in kilometres (for display only)."""

DAY: float = 1.0
"""Day (the unit of time)."""

DAY_SECONDS: float = 86_400.0
YEAR_DAYS: float = 365.25

ARCSEC: float = math.pi / (180.0 * 3600.0)
"""One arcsecond in radians."""

ARCMIN: float = 60.0 * ARCSEC
DEG: float = math.pi / 180.0
