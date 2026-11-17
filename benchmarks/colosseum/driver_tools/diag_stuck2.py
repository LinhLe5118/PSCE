import sys, time, hashlib
sys.path.insert(0,".")
from common import COHORTS, DATA, load_json, use_workspace
use_workspace()
sys.path[:0]=[str(__import__("pathlib").Path("../workspace/code").resolve())]
import numpy as np
from oeh import tabular, posterior
orig_pv=tabular.policy_values
calls={"opt":0}
def ov(P,r,gamma,*,max_iterations=10_000):
    actions=P.shape[-2]; pi=r.argmax(-1); t=time.time()
    for it in range(max_iterations):
        policy=np.eye(actions)[pi]; v=orig_pv(P,r,policy,gamma)
        q=r+gamma*np.einsum("...san,...n->...sa",P,v)
        cur=np.take_along_axis(q,pi[...,None],-1)[...,0]
        imp=q.max(-1)>cur+1e-12*(1.0+np.abs(cur))
        if not imp.any():
            calls["opt"]+=1; print(f"PI call {calls['opt']} shape {P.shape} iters {it} {time.time()-t:.1f}s",flush=True); return v,pi
        if it%500==0 and it: print(f"  ...iter {it}, still improving {int(imp.sum())} entries, {time.time()-t:.0f}s",flush=True)
        pi=np.where(imp,q.argmax(-1),pi)
    raise RuntimeError("no conv")
tabular.optimal_values=ov
from scripts.analysis import run_oeh_final_120 as R
doc=load_json(COHORTS["c7"]/"protocol.json")
row=[r for r in R.pilot_rows(DATA/"confirm_bundle") if r["instance_id"]=="confirm_colosseum_deepsea_002" and r["policy_family"]=="q_star_softmax"][0]
d=hashlib.sha256(f"external-colosseum-confirm-20261014|{row['instance_id']}|{row['policy_family']}".encode()).digest()
R._repo_first()
R.oeh_b_group({"row":row,"path":"E:/research/oeh_colosseum_trial/.tmp/diag_group.json","dataset":str(DATA/"confirm_bundle"),
 "structure":doc["oeh_final"]["structure"],"draws":doc["oeh_final"]["requirement_draws"],"seed":[int.from_bytes(d[:8],"little")],"backend":"cpu"})
