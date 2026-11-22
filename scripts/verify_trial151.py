"""Check the reported operating point against the Optuna study it came from.

`tab:selected_config` and the TUNED dict in run_stage5_bypass.py both claim to be trial
151 of the re-run NSGA-II search, normalised. This reads the study and verifies that,
rather than taking it on trust: that 151 is in fact the F1 argmax over the non-dominated
set, and that the normalised weights match what the manuscript prints.

    python scripts/verify_trial151.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PHD = Path("data")
DB = PHD / "Output_Repro/Tuning/Session_V5-3_Pareto_20260827_134728/optuna_pareto.db"
OUT = PHD / "Output_Repro/Tuning"

# The study samples raw weights under these names; the objective normalises them to
# sum to one before handing them to the matcher, and that is what the manuscript prints.
WEIGHT_KEYS = {"raw_w_shape": "shape", "raw_w_bear": "bearing",
               "raw_w_disp": "displacement", "raw_w_sinu": "sinuosity",
               "raw_w_topo": "topology"}


def main() -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name=None, storage=f"sqlite:///{DB}") \
        if len(optuna.study.get_all_study_names(f"sqlite:///{DB}")) == 1 else None
    if study is None:
        names = optuna.study.get_all_study_names(f"sqlite:///{DB}")
        raise SystemExit(f"expected one study, found {names}")

    done = [t for t in study.trials if t.values]
    print(f"study: {study.study_name}   complete trials: {len(done)}")

    # objectives are (recall, precision); F1 is the harmonic mean
    def f1(t):
        r, p = t.values[0], t.values[1]
        return 2 * r * p / (r + p) if (r + p) else 0.0

    front = study.best_trials
    print(f"non-dominated members: {len(front)}")
    best = max(front, key=f1)
    print(f"F1 argmax over the frontier: trial {best.number}  "
          f"R={best.values[0]:.4f} P={best.values[1]:.4f} F1={f1(best):.4f}")

    raw = {k: best.params[k] for k in WEIGHT_KEYS if k in best.params}
    tot = sum(raw.values())
    norm = {WEIGHT_KEYS[k]: round(v / tot, 4) for k, v in raw.items()}
    print("\nnormalised weights:")
    for k, v in norm.items():
        print(f"  {k:<22} {v:.4f}")
    other = {k: v for k, v in best.params.items() if k not in WEIGHT_KEYS}
    print("\nother sampled parameters:")
    for k, v in sorted(other.items()):
        print(f"  {k:<28} {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected_trial.json").write_text(json.dumps({
        "study": study.study_name, "complete_trials": len(done),
        "frontier_members": len(front),
        "selected_trial": best.number,
        "recall": round(best.values[0], 4), "precision": round(best.values[1], 4),
        "f1": round(f1(best), 4),
        "weights_normalised": norm, "params_raw": best.params,
    }, indent=2))
    print(f"\nwritten -> {OUT/'selected_trial.json'}")


if __name__ == "__main__":
    main()
