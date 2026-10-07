"""Inference server for the exact frozen Heavy-adapted Transformer V3."""
from __future__ import annotations
import argparse,hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
import torch
from docking_perception.geometry.se2_target import body_to_dock_pose,horizontal_covariance_to_dock
from docking_perception.integration import heavy_v3_protocol as P
from docking_perception.models.heads import decode_pose
from docking_perception.models.transformer import DockingTransformer
from docking_perception.training.data import Normalisation

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'models'
CHECKPOINT=OUT/'heavy_adapted_transformer_v3.pt';FREEZE=OUT/'SELECTION_FREEZE.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(device):
 f=json.loads(FREEZE.read_text())
 if sha(CHECKPOINT)!=f['checkpoint_sha256']:raise RuntimeError('V3 checkpoint hash mismatch')
 ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
 if ck.get('representation')!='dock_relative_se2_body_translation':raise RuntimeError('wrong V3 representation')
 m=DockingTransformer(with_uncertainty=True);m.load_state_dict(ck['state_dict']);m.to(device).eval()
 return m,Normalisation.from_dict(ck['normalisation']),np.asarray(ck['calibration_se2'],float),f['checkpoint_sha256']
def infer(m,norm,cal,camera,scan,device):
 im=torch.from_numpy(np.transpose(camera.astype(np.float32)/255,(2,0,1))).unsqueeze(0).to(device);sc=torch.from_numpy(scan.astype(np.float32)).unsqueeze(0).to(device)
 with torch.no_grad():mu,lv=m(im,sc)
 se2=norm.decode(decode_pose(mu).cpu().numpy())[0];pose=body_to_dock_pose(se2);sg=np.exp(.5*lv[0].cpu().numpy())*np.r_[norm.std,1.]*cal
 cov=horizontal_covariance_to_dock(se2,np.diag([sg[0]**2,sg[1]**2,sg[3]**2]));sigma=np.array([np.sqrt(max(cov[0,0],0)),np.sqrt(max(cov[1,1],0)),sg[2],np.sqrt(max(cov[2,2],0))])
 return pose,sigma
def serve(socket_path=P.DEFAULT_SOCKET,device=None):
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu');m,norm,cal,h=load(device)
 if os.path.exists(socket_path):os.unlink(socket_path)
 server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(socket_path);server.listen(1);lat=[];print(f'Heavy V3 perception ready checkpoint={h} device={device}',flush=True)
 try:
  while True:
   conn,_=server.accept()
   with conn:
    while True:
     try:payload=P.recv_exact(conn,P.REQUEST_BYTES)
     except ConnectionError:break
     camera,scan=P.unpack_request(payload);tic=time.perf_counter();pose,sigma=infer(m,norm,cal,camera,scan,device);lat.append(time.perf_counter()-tic);conn.sendall(P.pack_response(pose,sigma))
 finally:
  server.close()
  if os.path.exists(socket_path):os.unlink(socket_path)
  if lat:print(f'V3 frames={len(lat)} mean_ms={np.mean(lat)*1000:.2f} p95_ms={np.percentile(lat,95)*1000:.2f}',flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--socket',default=P.DEFAULT_SOCKET);a.add_argument('--device');q=a.parse_args();serve(q.socket,q.device)
if __name__=='__main__':main()
