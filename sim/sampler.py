from __future__ import annotations

import math
import numpy as np


def sample_object_spec(sampler_cfg: dict, rng: np.random.Generator) -> dict:
    # Zieht eine zufällige Form und sampelt alle Dimensionen, Masse und Reibung
    # gleichverteilt aus den in der Config definierten Ranges (Feix et al. 2014).
    shape = str(rng.choice(sampler_cfg["shapes"]))
    dims  = sampler_cfg[shape]

    spec: dict = {
        "shape":            shape,
        "mass_kg":          _uniform(sampler_cfg["mass_kg"], rng),
        "lateral_friction": _uniform(sampler_cfg["lateral_friction"], rng),
        "yaw_rad":          float(rng.uniform(0.0, 2.0 * math.pi)),  # zufällige Rotation um Z
    }
    for key, bounds in dims.items():
        spec[key] = _uniform(bounds, rng)

    if shape == "rect_cylinder":
        # thickness ist per Konvention immer die kürzere Seite, width die längere.
        t, w = spec["thickness_cm"], spec["width_cm"]
        spec["thickness_cm"] = min(t, w)
        spec["width_cm"]     = max(t, w)

    return spec


def _uniform(bounds: dict, rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds["min"], bounds["max"]))
