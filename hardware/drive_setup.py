# Einstiegspunkt: Calibration oder Pregrasp auswählen und starten.
# Zusätzliche CLI-Args werden an das aufgerufene Skript weitergereicht.
#
# Verwendung:
#   python -m hardware.drive_setup                  # interaktive Auswahl
#   python -m hardware.drive_setup --sim-only       # Args werden weitergereicht
#   python -m hardware.drive_setup calibration      # direkt Calibration
#   python -m hardware.drive_setup pregrasp         # direkt Pregrasp
from __future__ import annotations

import sys


def _ask_choice() -> str:
    print("\n" + "=" * 60)
    print("  AR10 Setup-Drive")
    print("=" * 60)
    print("  (1) Calibration  — EE-Offset kalibrieren")
    print("  (2) Pregrasp     — Greifpunkt anfahren + Lift testen")
    print("  (q) Beenden")
    print()
    while True:
        choice = input("Auswahl [1/2/q]: ").strip().lower()
        if choice in ("1", "cal", "calibration"):
            return "calibration"
        if choice in ("2", "pre", "pregrasp"):
            return "pregrasp"
        if choice in ("q", "quit", "exit"):
            return "quit"
        print("  Bitte 1, 2 oder q eingeben.")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in ("calibration", "pregrasp"):
        mode = args[0]
        forwarded = args[1:]
    else:
        mode = _ask_choice()
        forwarded = args

    if mode == "quit":
        print("[setup] Beendet.")
        return

    # sys.argv überschreiben damit das aufgerufene main() seine eigenen Args parst.
    sys.argv = [sys.argv[0]] + list(forwarded)

    if mode == "calibration":
        print("\n[setup] Starte Calibration ...")
        from hardware.drive_calibration import main as cal_main
        cal_main()
    elif mode == "pregrasp":
        print("\n[setup] Starte Pregrasp ...")
        from hardware.drive_pregrasp import main as pre_main
        pre_main()


if __name__ == "__main__":
    main()
