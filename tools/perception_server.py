#!/usr/bin/env python3
"""Serve a DEVELOPMENT checkpoint over the frozen V5 SE2 wire protocol.

The frozen server (`heavy_v5_perception_server`) hard-codes its checkpoint path
and refuses to start if the file's hash does not match SELECTION_FREEZE.json.
That guard is exactly right and is left alone; this is a separate entry point so
a candidate can be measured with the identical protocol, preprocessing and
decode path, without touching the frozen server or the frozen checkpoint.
"""
from __future__ import annotations
import argparse, os, socket, time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(os.environ.get("AUV_SIM_ROOT", "")).expanduser() / "src/uuv_sim_physics"))
import numpy as np, torch
from docking_perception.integration import heavy_v5_protocol as P
from docking_perception.models.heads import decode_pose
from docking_perception.models.transformer import DockingTransformer
from docking_perception.training.data import Normalisation

def load(path, device):
    """Build whichever architecture the checkpoint actually contains.

    The fair perception comparison serves a Transformer and two LateFusionMLP
    baselines over this same wire protocol, so the architecture is detected
    from the state_dict rather than assumed. A late-fusion checkpoint carries
    `fuse.0.weight`, whose shape gives both the hidden width and the feature
    dimension, so the capacity-matched variant (hidden=723) and the
    loss-matched one (hidden=256) load without a flag that could be set wrong.

    Before this, the server hard-coded DockingTransformer and both baseline
    evaluations died in load_state_dict -- the comparison silently reported
    only the Transformer.
    """
    from docking_perception.models.baselines import LateFusionMLP
    ck=torch.load(path, map_location="cpu", weights_only=False)
    sd=ck["state_dict"]
    if "fuse.0.weight" in sd:
        hidden, twice_dim = sd["fuse.0.weight"].shape
        m=LateFusionMLP(twice_dim//2, hidden, with_uncertainty=True)
        arch=f"late_fusion(hidden={hidden})"
    else:
        m=DockingTransformer(with_uncertainty=True)
        arch="transformer"
    m.load_state_dict(sd); m.to(device).eval()
    print(f"architecture detected: {arch}", flush=True)
    return (m, Normalisation.from_dict(ck["normalisation"]),
            np.asarray(ck["calibration_se2"], float))

#: FAULT INJECTION, validation only. Adds a constant offset to the network's
#: lateral output after a delay, simulating a visual channel that develops a
#: bias in flight. A bias present from the first frame is unobservable -- the
#: filter seeds FROM the visual, so the offset is inherited and produces no
#: innovation at all -- which is why the fault that must be contained is a step
#: appearing after seeding. Unset, this does nothing.
_BIAS_M = float(os.environ.get("DOCK_INJECT_LATERAL_BIAS_M", "0") or 0)
_BIAS_AFTER_S = float(os.environ.get("DOCK_INJECT_AFTER_S", "0") or 0)
_T0 = None


def infer(m, norm, cal, camera, scan, device):
    im=torch.from_numpy(np.transpose(camera.astype(np.float32)/255,(2,0,1))).unsqueeze(0).to(device)
    sc=torch.from_numpy(scan.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad(): mu,lv=m(im,sc)
    se2=norm.decode(decode_pose(mu).cpu().numpy())[0]
    sigma=np.exp(.5*lv[0].cpu().numpy())*np.r_[norm.std,1.]*cal
    global _T0
    if _BIAS_M:
        if _T0 is None: _T0 = time.monotonic()
        if time.monotonic() - _T0 >= _BIAS_AFTER_S:
            se2 = se2.copy(); se2[1] = se2[1] + _BIAS_M
    return se2,sigma

def main():
    a=argparse.ArgumentParser(); a.add_argument("--socket",required=True)
    a.add_argument("--checkpoint",required=True); a.add_argument("--device",default="cpu")
    q=a.parse_args()
    m,norm,cal=load(q.checkpoint,q.device)
    if os.path.exists(q.socket): os.unlink(q.socket)
    s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(q.socket); s.listen(1)
    print(f"DEV server ready checkpoint={q.checkpoint}",flush=True)
    try:
        while True:
            conn,_=s.accept()
            with conn:
                while True:
                    try: payload=P.recv_exact(conn,P.REQUEST_BYTES)
                    except ConnectionError: break
                    cam,scan=P.unpack_request(payload)
                    se2,sigma=infer(m,norm,cal,cam,scan,q.device)
                    conn.sendall(P.pack_response(se2,sigma))
    finally:
        s.close()
        if os.path.exists(q.socket): os.unlink(q.socket)

if __name__=="__main__": main()
