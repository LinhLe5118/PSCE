"""Record the comparator extension of the confirmatory run before computing it."""
import datetime, hashlib, json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
R = root / "results" / "confirm"
out = R / "protocol_colosseum_amendment_1_comparators.json"
if out.exists():
    raise SystemExit("amendment already frozen")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
doc = {
    "name": "confirmatory Colosseum run: extension to the cohort-7 comparator set",
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status_at_freeze": "learner outcomes of the 72 MDPs already observed and scored (primary verdict GO/GO)",
    "post_hoc": True,
    "why": "the paper's main result (cohort 7) gates OEH against 11 comparators (tier 1 shared information, "
           "tier 2 privileged); the confirmatory run only had B2 and B2+hardness",
    "fixed_by_team_protocol": "comparator definitions, fields and fitting are the team's cohort-7 ones "
                              "(run_oeh_final_120.FIELD_SETS / all_forecasts); nothing is chosen or tuned here",
    "unchanged": ["72 MDPs, pilots, labels", "sealed OEH predictions (must be reproduced bit-for-bit by all_forecasts)",
                  "meta-training panel (720 MDPs)"],
    "new_inputs": {
        "published": "oeh_final_published (TQ/SACo reference, ERI, BWD) on the Colosseum kernels",
        "tq_saco_original": "authors' code ml-jku/OfflineRL commit 8cc8eec via oeh_final_tqsaco_original",
        "eh_oracle": "finite_horizon_online_bounds on the Colosseum kernels",
    },
    "gate": "same as cohort 7: 22 joint contrasts (11 comparators x Brier, log loss), 10,000 whole-MDP bootstrap, "
            "simultaneous one-sided 95% lower bounds > 0; verdicts tier1, tier2, all",
    "secondary": "finite-case analysis: conditional budget forecast among non-censored labels (descriptive)",
    "sealed_inputs_sha256": {n: sha(R / n) for n in ("protocol_colosseum.json", "colosseum_predictions_preoutcome.csv",
                                                      "colosseum_preoutcome_seal.json", "colosseum_labels.csv")},
}
out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
print("frozen", out.name, sha(out))
