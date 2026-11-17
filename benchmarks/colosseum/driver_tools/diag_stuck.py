import sys, faulthandler, hashlib, json
sys.argv=["x"]; sys.path.insert(0,".")
from common import COHORTS, DATA, load_json, use_workspace
use_workspace()
from scripts.analysis import run_oeh_final_120 as R
doc=load_json(COHORTS["c7"]/"protocol.json")
rows=[r for r in R.pilot_rows(DATA/"confirm_bundle") if r["instance_id"]=="confirm_colosseum_deepsea_002" and r["policy_family"]=="q_star_softmax"]
row=rows[0]
digest=hashlib.sha256(f"external-colosseum-confirm-20261014|{row['instance_id']}|{row['policy_family']}".encode()).digest()
task={"row":row,"path":"E:/research/oeh_colosseum_trial/.tmp/diag_group.json","dataset":str(DATA/"confirm_bundle"),
      "structure":doc["oeh_final"]["structure"],"draws":doc["oeh_final"]["requirement_draws"],
      "seed":[int.from_bytes(digest[:8],"little")],"backend":"cpu"}
R._repo_first()
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
R.oeh_b_group(task)
