from __future__ import annotations


def step_reward(
    n_contact_prev:   int,
    n_contact_now:    int,
    n_target:         int,
    pedestal_contact: bool,
    cfg:              dict,
) -> float:
    # Zeit-Penalty + Bonus bei jedem NEUEN Kontakt (potential-based) + kleiner Halt-Bonus.
    # Der Δ-Term gibt ein scharfes Lernsignal im Moment des Erst-Kontakts.
    # w_hold belohnt das Aufrechterhalten des Kontakts, sodass die Policy nicht abrupt loslässt.
    r = cfg["r_step"]
    capped_now    = min(n_contact_now,  n_target)
    capped_prev   = min(n_contact_prev, n_target)
    new_contacts  = max(0, capped_now - capped_prev)
    r += cfg["w_contact"] * new_contacts / n_target
    r += cfg.get("w_hold", 0.0) * capped_now / n_target
    if pedestal_contact:
        r -= cfg["w_pedestal"]
    return r


def terminal_reward(lifted: bool, cfg: dict) -> float:
    # Wird einmalig am Episodenende aufgerufen, nach dem Lift-Test.
    return cfg["r_lift_success"] if lifted else cfg["r_lift_fail"]
