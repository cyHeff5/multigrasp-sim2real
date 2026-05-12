from __future__ import annotations


def build_grid(sampler_cfg: dict, step_cm: float = 1.0) -> list[dict]:
    # Systematisches Raster über alle Shapes aus dem Sampler-Config.
    # Gibt eine flache Liste von obj_specs zurück, einen pro Gitterpunkt.
    shapes = sampler_cfg.get("shapes", [])
    specs: list[dict] = []
    for shape in shapes:
        specs.extend(_grid_for_shape(shape, sampler_cfg, step_cm))
    return specs


def _grid_for_shape(shape: str, cfg: dict, step_cm: float) -> list[dict]:
    base = {
        "shape":   shape,
        "yaw_rad": 0.0,
    }

    if shape in ("sphere", "cube"):
        lo, hi = _bounds(cfg[shape]["size_cm"])
        return [{**base, "size_cm": s} for s in _arange(lo, hi, step_cm)]

    if shape == "cylinder":
        lo, hi    = _bounds(cfg["cylinder"]["thickness_cm"])
        height_cm = _mid(cfg["cylinder"]["height_cm"])
        return [
            {**base, "thickness_cm": t, "height_cm": height_cm}
            for t in _arange(lo, hi, step_cm)
        ]

    if shape == "rect_cylinder":
        # 2D-Raster: thickness × width.
        t_lo, t_hi = _bounds(cfg["rect_cylinder"]["thickness_cm"])
        w_lo, w_hi = _bounds(cfg["rect_cylinder"]["width_cm"])
        height_cm  = _mid(cfg["rect_cylinder"]["height_cm"])
        return [
            {**base, "thickness_cm": t, "width_cm": w, "height_cm": height_cm}
            for t in _arange(t_lo, t_hi, step_cm)
            for w in _arange(w_lo, w_hi, step_cm)
        ]

    return []


# Helpers
def _mid(bounds: dict) -> float:
    return 0.5 * (float(bounds["min"]) + float(bounds["max"]))


def _bounds(b: dict) -> tuple[float, float]:
    return float(b["min"]), float(b["max"])


def _arange(lo: float, hi: float, step: float) -> list[float]:
    # lo + i*step statt kumulativer Addition um Gleitkomma-Drift zu vermeiden.
    n = round((hi - lo) / step)
    return [round(lo + i * step, 4) for i in range(n + 1)]


def spec_label(spec: dict) -> str:
    # Lesbare Kurzbezeichnung für einen obj_spec, wird in der Eval-Ausgabe verwendet.
    shape = spec["shape"]
    if shape in ("sphere", "cube"):
        return f"{shape:<14} size={spec['size_cm']:.1f}cm"
    if shape == "cylinder":
        return f"cylinder       d={spec['thickness_cm']:.1f}cm  h={spec['height_cm']:.1f}cm"
    if shape == "rect_cylinder":
        return (f"rect_cylinder  t={spec['thickness_cm']:.1f}cm "
                f"w={spec['width_cm']:.1f}cm  h={spec['height_cm']:.1f}cm")
    return str(spec)
