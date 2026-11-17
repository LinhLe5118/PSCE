"""Freeze the confirmatory Colosseum protocol before any new MDP is generated."""
import datetime
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
out = root / "results" / "confirm" / "protocol_colosseum.json"
if out.exists():
    raise SystemExit("confirmatory protocol already frozen")
explore = root / "results"
doc = {
    "name": "confirmatory Colosseum test of OEH current (fresh MDPs)",
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "new_mdps_generated": 0, "learner_outcomes_observed": 0,
    "motivation": "exploratory run (32 MDPs) gave OEH vs B2 Brier GO and log-loss lower bound -0.021; "
                  "its effect (mean log-loss gain 0.089, MDP-level SE about 0.062) implies about 50 MDPs "
                  "for a positive simultaneous bound; 72 are used",
    "exploratory_evidence_sha256": {name: hashlib.sha256((explore / name).read_bytes()).hexdigest()
                                    for name in ("protocol_colosseum.json", "protocol_colosseum_amendment_1.json",
                                                 "colosseum_predictions_preoutcome.csv", "colosseum_labels.csv")},
    "mdps": {
        "source": "Colosseum (Conserva & Rauber, NeurIPS 2022), authors' sample_mdp_parameters, continuous setting",
        "families": ["riverswim", "deepsea", "frozenlake", "simplegrid"],
        "sampler_seed": 2027, "draws_per_family": 600, "export_cap_per_family": 30,
        "filters": ["stochastic kernel", "16 <= |S||A| <= 256", "project dynamic-range filter"],
        "exclusion": "any kernel identical to the 32 exploratory MDPs",
        "selection": "first 18 per family in sampler order -> 72 MDPs (fewer only if a family runs out; reported)",
        "data_seed_base": 9_500_000,
    },
    "unchanged_from_exploratory": [
        "project generator, 5 behaviors, pilot x10, 10 training seeds, BC/CFQI/IQL hyper-parameters, label rule",
        "forecaster: sealed seven-cohort meta-training via expanded_meta; no refit choices, no tuning",
        "comparators: oeh_final (108), b2_matched (18), b2_plus_colosseum (B2 + log diameter, value norm, log gaps)",
        "team's documented expectile fix applied only where the archived IQL raises",
        "metrics, MDP-level aggregation, 10,000 whole-MDP bootstrap, simultaneous max-t one-sided 95% bounds",
    ],
    "gate": {
        "primary": "OEH vs b2_matched: mean gain > 0 and simultaneous lower bound > 0 for BOTH integrated Brier "
                   "and seven-class log loss",
        "secondary": "OEH vs b2_plus_colosseum: same rule",
        "no_minimum_relative_gain": True,
    },
    "reporting": "report GO or NO-GO as computed; descriptive analyses (family, learner, finite vs censored, "
                 "posterior-draw failures) do not change the verdict; no re-run on these MDPs after outcomes",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
print("frozen", out, hashlib.sha256(out.read_bytes()).hexdigest())
