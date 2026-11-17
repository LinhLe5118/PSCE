import sys, time, hashlib
sys.path.insert(0,".")
from common import COHORTS, DATA, load_json, use_workspace
use_workspace()
sys.path[:0]=[str(__import__("pathlib").Path("../workspace/code").resolve())]
import numpy as np
from oeh import tabular, posterior
cap={}
orig=posterior._draw
def dr(*a,**k):
    out=orig(*a,**k); cap["P"]=out[0]; cap["R"]=out[1]; raise KeyboardInterrupt
posterior._draw=dr
from scripts.analysis import run_oeh_final_120 as R
doc=load_json(COHORTS["c7"]/"protocol.json")
row=[r for r in R.pilot_rows(DATA/"confirm_bundle") if r["instance_id"]=="confirm_colosseum_deepsea_002" and r["policy_family"]=="q_star_softmax"][0]
d=hashlib.sha256(f"external-colosseum-confirm-20261014|{row['instance_id']}|{row['policy_family']}".encode()).digest()
R._repo_first()
try:
    R.oeh_b_group({"row":row,"path":"E:/research/oeh_colosseum_trial/.tmp/diag_group.json","dataset":str(DATA/"confirm_bundle"),
     "structure":doc["oeh_final"]["structure"],"draws":doc["oeh_final"]["requirement_draws"],"seed":[int.from_bytes(d[:8],"little")],"backend":"cpu"})
except KeyboardInterrupt: pass
P=cap["P"]; print("P",P.shape,P.dtype)
tiny=np.finfo(float).tiny
sub=(P!=0)&(np.abs(P)<tiny); print("subnormal entries",int(sub.sum()),"of",P.size, "min nonzero",P[P>0].min())
print("nan",np.isnan(P).sum())
r=np.einsum("bsan,bsan->bsa",P,cap["R"]); pol=np.eye(2)[r.argmax(-1)]
for name,PP in (("raw",P),("flushed",np.where(sub,0.0,P))):
    t=time.time(); tabular.policy_values(PP,r,pol,0.95); print(name,"solve %.3fs"%(time.time()-t))
