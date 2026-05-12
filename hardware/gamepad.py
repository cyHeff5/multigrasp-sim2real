# Gamepad- und Tastatur-Input.
# read_inputs() gibt immer dasselbe Dict zurück.
from __future__ import annotations

# Totzone für Analogsticks: Werte darunter werden als 0 behandelt (Drift-Unterdrückung).
_DEAD = 0.08


def _btn(js, i: int) -> bool:
    try:
        return bool(js.get_numbuttons() > i and js.get_button(i))
    except Exception:
        return False


def _axis(js, i: int) -> float:
    try:
        v = float(js.get_axis(i))
        return v if abs(v) > _DEAD else 0.0
    except Exception:
        return 0.0


def read_inputs(js) -> dict:
    # js=None -> Tastatur-Fallback (SPACE=A, BACKSPACE=B, X/Y, A/D=LB/RB, Pfeile=Stick).
    import pygame
    pygame.event.pump()
    if js is not None:
        return {
            "a":    _btn(js, 0),
            "b":    _btn(js, 1),
            "x":    _btn(js, 2),
            "y":    _btn(js, 3),
            "rb":   _btn(js, 5),
            "lb":   _btn(js, 4),
            "menu": _btn(js, 7),
            "sx":   _axis(js, 0),
            "sy":  -_axis(js, 1),  # Joystick-Konvention: oben = negativ -> invertieren.
        }
    else:
        keys = pygame.key.get_pressed()
        return {
            "a":    bool(keys[pygame.K_SPACE]),
            "b":    bool(keys[pygame.K_BACKSPACE]),
            "x":    bool(keys[pygame.K_x]),
            "y":    bool(keys[pygame.K_y]),
            "rb":   bool(keys[pygame.K_d]),
            "lb":   bool(keys[pygame.K_a]),
            "menu": bool(keys[pygame.K_ESCAPE]),
            "sx":   (-1.0 if keys[pygame.K_LEFT]  else (1.0 if keys[pygame.K_RIGHT] else 0.0)),
            "sy":   (-1.0 if keys[pygame.K_DOWN]  else (1.0 if keys[pygame.K_UP]    else 0.0)),
        }


def init_pygame_joystick():
    # Gibt den ersten gefundenen Joystick zurück, oder None wenn keiner angeschlossen ist.
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"[gamepad] {js.get_name()}")
        return js
    print("[gamepad] Kein Gamepad gefunden — Tastatur-Fallback aktiv.")
    return None
