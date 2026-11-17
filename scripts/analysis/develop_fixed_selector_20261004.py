"""Development-only grouped OOF assessment of a router against fixed CFQI."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from scripts.analysis.oeh_final_analysis import simultaneous_lower

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"


def design(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pilot = pd.read_csv(out / "pilot_features.csv")
    pilot = pilot[pilot.algorithm == "cfqi_tabular"].drop(columns="algorithm")
    pred = pd.read_csv(out / "predictions_preoutcome.csv")
    pred = pred[(pred.method == "oeh_final") & (pred.algorithm == "cfqi_tabular")]
    pred = pred[["mdp_id", "policy_family", *[f"p_{i}" for i in range(7)]]]
    frame = pilot.merge(pred, on=["mdp_id", "policy_family"], validate="one_to_one")
    frame["family"] = frame.mdp_id.str.extract(r"_(uniform_garnet|forward_layered|bottleneck|lock_structured)_\d+$")[0]
    x = frame.drop(columns=["mdp_id", "policy_family", "family"]).select_dtypes(include=[np.number])
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    x = pd.concat([x, pd.get_dummies(frame.family, prefix="family")], axis=1)
    return frame[["mdp_id", "policy_family"]], x


def model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=150, min_samples_leaf=7,
                               max_features=.8, n_jobs=4, random_state=42)


def main() -> None:
    key, x = design(OUT)
    y = pd.read_csv(OUT / "selection_group_outcomes.csv")
    keep = key.merge(y[["mdp_id", "policy_family"]], on=["mdp_id", "policy_family"], validate="one_to_one")
    positions = pd.MultiIndex.from_frame(key).get_indexer(pd.MultiIndex.from_frame(keep))
    key, x = key.iloc[positions].reset_index(drop=True), x.iloc[positions].reset_index(drop=True)
    frame = key.merge(y, on=["mdp_id", "policy_family"], validate="one_to_one")
    names = [c[6:-4] for c in frame if c.startswith("fixed:") and c.endswith("|suc")]
    success = frame[[f"fixed:{name}|suc" for name in names]].to_numpy()
    ret = frame[[f"fixed:{name}|ret" for name in names]].to_numpy()
    oof_s = np.zeros_like(success)
    oof_r = np.zeros_like(ret)
    for fit, val in GroupKFold(5).split(x, groups=key.mdp_id):
        oof_s[val] = model().fit(x.iloc[fit], success[fit]).predict(x.iloc[val])
        oof_r[val] = model().fit(x.iloc[fit], ret[fit]).predict(x.iloc[val])
    cfqi = names.index("cfqi_tabular")
    for floor in (0, .02, .05, .1):
        score = oof_s + .15 * oof_r
        choice = score.argmax(1)
        choice = np.where(score[np.arange(len(choice)), choice] - score[:, cfqi] > floor, choice, cfqi)
        effect = pd.DataFrame({"mdp_id": key.mdp_id,
                               "success": success[np.arange(len(choice)), choice] - success[:, cfqi],
                               "return": ret[np.arange(len(choice)), choice] - ret[:, cfqi]})
        mdp = effect.groupby("mdp_id")[["success", "return"]].mean().to_numpy()
        bound, _ = simultaneous_lower(mdp, 10000, 20261004)
        print(f"floor={floor} switches={(choice != cfqi).sum()} effect={mdp.mean(0)} lower={bound}", flush=True)


if __name__ == "__main__":
    main()
