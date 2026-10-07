"""Frozen V3 inference server exposing native SE(2) outputs for V5."""
from __future__ import annotations
import argparse,os,socket
import numpy as np,torch
from docking_perception.integration import heavy_v5_protocol as P
from docking_perception.integration.heavy_v3_perception_server import load
from docking_perception.models.heads import decode_pose
def infer(m,norm,cal,camera,scan,device):
 im=torch.from_numpy(np.transpose(camera.astype(np.float32)/255,(2,0,1))).unsqueeze(0).to(device);sc=torch.from_numpy(scan.astype(np.float32)).unsqueeze(0).to(device)
 with torch.no_grad():mu,lv=m(im,sc)
 se2=norm.decode(decode_pose(mu).cpu().numpy())[0];sigma=np.exp(.5*lv[0].cpu().numpy())*np.r_[norm.std,1.]*cal
 return se2,sigma
def serve(socket_path=P.DEFAULT_SOCKET,device=None):
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu');m,norm,cal,h=load(device)
 if os.path.exists(socket_path):os.unlink(socket_path)
 server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(socket_path);server.listen(1);print(f'Heavy V5 SE2 server ready checkpoint={h}',flush=True)
 try:
  while True:
   conn,_=server.accept()
   with conn:
    while True:
     try:payload=P.recv_exact(conn,P.REQUEST_BYTES)
     except ConnectionError:break
     camera,scan=P.unpack_request(payload);se2,sigma=infer(m,norm,cal,camera,scan,device);conn.sendall(P.pack_response(se2,sigma))
 finally:
  server.close()
  if os.path.exists(socket_path):os.unlink(socket_path)
def main():
 a=argparse.ArgumentParser();a.add_argument('--socket',default=P.DEFAULT_SOCKET);a.add_argument('--device');q=a.parse_args();serve(q.socket,q.device)
if __name__=='__main__':main()
