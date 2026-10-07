"""V4 observable-only overlap fusion of frozen V3 fixes and DVL propagation."""
from __future__ import annotations
import math,socket,time
import numpy as np
from docking_perception.integration.heavy_bridge import HeavyObservationBridge,oculus_image_to_network_v2
from docking_perception.integration import heavy_v3_protocol as P
from uuv_sim_physics.sensors.oculus_m750d import OculusM750dRenderer,extract_observable_features

class HeavyV4TerminalProvider:
 hardware_realisable=True;requires=('camera','realistic_fls','dvl','imu','depth')
 NIS_LIMIT=9.21034;SIGMA_LIMIT_M=.10;DVL_STALE_S=.30;PROCESS_VELOCITY_MPS=.002;DVL_NOISE_MPS=.0005
 def __init__(self,socket_path=P.DEFAULT_SOCKET,connect_timeout_s=60.):self.socket_path=socket_path;self.connect_timeout_s=connect_timeout_s;self.bridge=HeavyObservationBridge();self._sock=None
 def reset(self,seed):
  self.close();self.renderer=OculusM750dRenderer(seed=int(seed));self.x=np.zeros(2);self.P=np.eye(2)*1e6;self.have=False;self.vel=np.zeros(2);self.last_dvl_stamp=None;self.dvl_age=float('inf');self.absolute_misses=0;self.safe=False;self.recovery_live=0
 def close(self):
  if self._sock is not None:self._sock.close();self._sock=None
 def _connect(self):
  deadline=time.monotonic()+self.connect_timeout_s;s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
  while True:
   try:s.connect(self.socket_path);self._sock=s;return
   except (FileNotFoundError,ConnectionRefusedError):
    if time.monotonic()>=deadline:s.close();raise RuntimeError('V3 perception server unavailable')
    time.sleep(.05)
 def _predict_dvl(self,obs):
  self.dvl_age+=obs.dt;fresh=obs.dvl_acquisition_t is not None and (self.last_dvl_stamp is None or obs.dvl_acquisition_t>self.last_dvl_stamp)
  if fresh:self.last_dvl_stamp=obs.dvl_acquisition_t
  if fresh and obs.dvl_valid and obs.dvl_velocity is not None:
   v=np.asarray(obs.dvl_velocity,float);c,s=math.cos(obs.yaw),math.sin(obs.yaw);self.vel=np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]]);self.dvl_age=0.;self.recovery_live+=1
  elif fresh:self.recovery_live=0
  live=self.dvl_age<=self.DVL_STALE_S
  if self.have and live:
   self.x=self.x+self.vel*obs.dt;q=((self.DVL_NOISE_MPS*obs.dt)**2+self.PROCESS_VELOCITY_MPS**2*obs.dt)*np.eye(2);self.P=self.P+q
  return live,fresh
 def _observe(self,obs):
  if obs.image is None or obs.fls_ranges is None:return None,'no_synchronized_observation'
  fan=np.asarray(obs.fls_ranges,np.float32)
  if fan.size!=512*9:raise ValueError(f'V4 requires 512x9 Oculus intersections, got {fan.size}')
  acoustic=self.renderer.render(fan.reshape(9,512),acquisition_t=float(obs.fls_acquisition_t or obs.t));support=extract_observable_features(acoustic)
  if not support['valid']:return None,'no_oculus_support'
  scan=oculus_image_to_network_v2(acoustic);camera=self.bridge.camera_uint8(obs.image)
  if self._sock is None:self._connect()
  self._sock.sendall(P.pack_request(camera,scan));pose,sigma=P.unpack_response(P.recv_exact(self._sock,P.RESPONSE_BYTES));z=np.asarray(pose[:2],float);sg=np.asarray(sigma[:2],float)
  if not np.isfinite(z).all() or not np.isfinite(sg).all():return None,'nonfinite_v3'
  if np.any(sg>self.SIGMA_LIMIT_M):return None,'v3_sigma_above_limit'
  return (z,sg,support),'candidate'
 def update(self,obs):
  from uuv_sim_physics.docking.state_providers import ProviderEstimate
  dvl_live,_=self._predict_dvl(obs);candidate,reason=self._observe(obs);accepted=False;nis=None;sigma=None
  if candidate is not None:
   z,sg,support=candidate;R=np.diag(sg**2);sigma=sg.tolist()
   if not self.have:self.x=z.copy();self.P=R.copy();self.have=True;accepted=True;nis=0.
   else:
    innovation=z-self.x;S=self.P+R;nis=float(innovation@np.linalg.solve(S,innovation))
    if nis<=self.NIS_LIMIT:
     K=self.P@np.linalg.inv(S);self.x=self.x+K@innovation;I=np.eye(2);self.P=(I-K)@self.P@(I-K).T+K@R@K.T;accepted=True
   reason='accepted' if accepted else 'innovation_rejected'
  self.absolute_misses=0 if accepted else self.absolute_misses+1
  if self.safe:
   if accepted or (dvl_live and self.recovery_live>=3):self.safe=False
  elif self.absolute_misses>=3 and not dvl_live:self.safe=True;self.recovery_live=0
  if not self.have:return ProviderEstimate(self.x.copy(),False,{'source':'v4','reason':reason,'dvl_age_s':self.dvl_age},velocity_xy=self.vel.copy(),mode='ACQUIRE')
  if self.safe:mode='SAFE_HOLD'
  elif accepted:mode='OVERLAP' if self.x[0]<=.8 and dvl_live else 'TRACK'
  elif dvl_live:mode='DONE' if self.x[0]<=.05 else 'COMMIT'
  else:mode='TRACK'
  detail={'source':'v4','visual_update':reason,'visual_accepted':accepted,'innovation_nis':nis,'reported_sigma_xy':sigma,'filter_sigma_xy':np.sqrt(np.maximum(np.diag(self.P),0)).tolist(),'dvl_age_s':self.dvl_age,'dvl_live':dvl_live,'dvl_locked':obs.dvl_beams_locked,'absolute_misses':self.absolute_misses}
  return ProviderEstimate(self.x.copy(),not self.safe,detail,velocity_xy=self.vel.copy(),mode=mode)
