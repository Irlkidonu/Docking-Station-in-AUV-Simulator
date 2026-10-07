"""Truth-free V3 wire protocol: rectified camera plus Oculus network tensor."""
from __future__ import annotations
import socket,struct
import numpy as np

CAMERA_SHAPE=(120,160,3); SCAN_SHAPE=(3,128)
DEFAULT_SOCKET="/tmp/docking_station_heavy_transformer_v3.sock"
_CB=int(np.prod(CAMERA_SHAPE));_SB=int(np.prod(SCAN_SHAPE))*4
REQUEST_BYTES=_CB+_SB;_RESPONSE=struct.Struct('<8f');RESPONSE_BYTES=_RESPONSE.size
def pack_request(camera,scan):
 if camera.shape!=CAMERA_SHAPE or camera.dtype!=np.uint8:raise ValueError('invalid V3 camera tensor')
 if scan.shape!=SCAN_SHAPE:raise ValueError('invalid V3 Oculus tensor')
 return np.ascontiguousarray(camera).tobytes()+np.ascontiguousarray(scan,np.float32).tobytes()
def unpack_request(payload):
 if len(payload)!=REQUEST_BYTES:raise ValueError('invalid V3 request size')
 return (np.frombuffer(payload,np.uint8,_CB).reshape(CAMERA_SHAPE),np.frombuffer(payload,np.float32,3*128,_CB).reshape(SCAN_SHAPE).copy())
def pack_response(pose,sigma):return _RESPONSE.pack(*np.r_[np.asarray(pose,np.float32).reshape(4),np.asarray(sigma,np.float32).reshape(4)].tolist())
def unpack_response(payload):
 if len(payload)!=RESPONSE_BYTES:raise ValueError('invalid V3 response size')
 x=np.asarray(_RESPONSE.unpack(payload),float);return x[:4],x[4:]
def recv_exact(sock:socket.socket,count:int):
 chunks=[]
 while count:
  b=sock.recv(count)
  if not b:raise ConnectionError('V3 socket closed')
  chunks.append(b);count-=len(b)
 return b''.join(chunks)
