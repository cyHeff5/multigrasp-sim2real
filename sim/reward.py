from __future__ import annotations


def step_reward(
    n_contact:        int,
    n_target:         int,
    pedestal_contact: bool,
    cfg:              dict,
) -> float:
    # Zeit-Penalty + linearer Kontakt-Bonus bis n_target, dann gedeckelt.
    # Gedeckelt damit die Policy nicht sequentiell Finger schliesst um mehr Reward zu sammeln.
    # pedestal_contact: True wenn ein Fingertip das Podest beruehrt.
    r = cfg["r_step"]
    r += cfg["w_contact"] * min(n_contact, n_target) / n_target
    if pedestal_contact:
        r -= cfg["w_pedestal"]
    return r


def terminal_reward(lifted: bool, cfg: dict) -> float:
    # Wird einmalig am Episodenende aufgerufen, nach dem Lift-Test.
    return cfg["r_lift_success"] if lifted else cfg["r_lift_fail"]
