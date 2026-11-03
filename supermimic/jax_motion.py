"""Host-load-once, device-resident motion tables matching origin MotionLib."""

from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from .jax_geometry import JAXG1Kinematics, mat_to_6d, quat_to_mat, so3_log, yaw_mat

CONTROL_DT=0.02
BIN_FRAMES=50


def _gradient(x):
    g=jnp.zeros_like(x)
    g=g.at[1:-1].set((x[2:]-x[:-2])/(2*CONTROL_DT))
    g=g.at[0].set((x[1]-x[0])/CONTROL_DT)
    g=g.at[-1].set((x[-1]-x[-2])/CONTROL_DT)
    return g


def _quat_slerp(q0,q1,t):
    dot=jnp.sum(q0*q1,-1,keepdims=True); q1=jnp.where(dot<0,-q1,q1); dot=jnp.clip(jnp.abs(dot),max=1.0); theta=jnp.arccos(dot); s=jnp.sin(theta); t=t[...,None]
    w0=jnp.where(s<1e-6,1-t,jnp.sin((1-t)*theta)/(s+1e-12)); w1=jnp.where(s<1e-6,t,jnp.sin(t*theta)/(s+1e-12)); out=w0*q0+w1*q1
    return out/jnp.maximum(jnp.linalg.norm(out,axis=-1,keepdims=True),1e-12)


def _assemble(obj,names,ps,Rs,qs,kin):
    lens=[int(x.shape[0]) for x in qs]; vs=[_gradient(x) for x in ps]; qds=[_gradient(x) for x in qs]; ws=[]; body=[]
    for p,R,q,v,qd in zip(ps,Rs,qs,vs,qds):
        w=jnp.zeros_like(p)
        if p.shape[0]>2:
            w=w.at[1:-1].set(so3_log(R[2:]@jnp.swapaxes(R[:-2],-1,-2))/(2*CONTROL_DT)); w=w.at[0].set(so3_log(R[1]@R[0].T)/CONTROL_DT); w=w.at[-1].set(so3_log(R[-1]@R[-2].T)/CONTROL_DT)
        ws.append(w); body.append(kin.forward(p,R,v,w,q,qd))
    obj.names=names; obj.clip_len=jnp.asarray(lens,jnp.int32); obj.clip_start=jnp.concatenate((jnp.zeros((1,),jnp.int32),jnp.cumsum(obj.clip_len[:-1]))); obj.clip_end=obj.clip_start+obj.clip_len-1; obj.num_frames=int(sum(lens)); obj.num_clips=len(lens); obj.clip_id=jnp.concatenate([jnp.full((n,),i,jnp.int32) for i,n in enumerate(lens)])
    cat=lambda xs:jnp.concatenate(xs,0); obj.root_p,obj.root_R,obj.root_v,obj.root_w=cat(ps),cat(Rs),cat(vs),cat(ws); obj.q,obj.qd=cat(qs),cat(qds); obj.body_p,obj.body_R,obj.body_v,obj.body_w=(cat([b[i] for b in body]) for i in range(4)); Rt=jnp.swapaxes(obj.root_R,-1,-2); obj.root_v_local=jnp.einsum('fij,fj->fi',Rt,obj.root_v); obj.root_w_local=jnp.einsum('fij,fj->fi',Rt,obj.root_w)
    starts=[]; clips=[]
    for c,n in enumerate(lens): starts.extend(range(int(obj.clip_start[c]),int(obj.clip_end[c])+1,BIN_FRAMES)); clips.extend([c]*int(np.ceil(n/BIN_FRAMES)))
    obj.bin_start=jnp.asarray(starts,jnp.int32); obj.bin_clip=jnp.asarray(clips,jnp.int32); obj.num_bins=len(starts); obj.frame_bin=jnp.zeros((obj.num_frames,),jnp.int32)
    for b,(s,c) in enumerate(zip(starts,clips)):
        e=min(s+BIN_FRAMES,int(obj.clip_end[c])+1); obj.frame_bin=obj.frame_bin.at[s:e].set(b)


class JAXMotion:
    def __init__(self,path: str|Path,kin: JAXG1Kinematics):
        files=sorted(glob.glob(str(Path(path)/"*.npz")))
        if not files: raise FileNotFoundError(f"no NPZ motions found in {path}")
        names=[]; ps=[]; Rs=[]; qs=[]
        for filename in files:
            with np.load(filename) as raw:
                if int(np.asarray(raw['fps']).reshape(-1)[0])!=50: raise ValueError(f"{filename} is not a 50 Hz clip")
                p=jnp.asarray(np.array(raw['body_pos_w'][:,1],dtype=np.float32,copy=True)); R=quat_to_mat(jnp.asarray(np.array(raw['body_quat_w'][:,1],dtype=np.float32,copy=True))); q=jnp.asarray(np.array(raw['joint_pos'],dtype=np.float32,copy=True))
            names.append(Path(filename).stem); ps.append(p); Rs.append(R); qs.append(q)
        _assemble(self,names,ps,Rs,qs,kin)

    @classmethod
    def from_csv_dir(cls,path: str|Path,kin: JAXG1Kinematics,src_fps: float=30.0):
        files=sorted(glob.glob(str(Path(path)/'*.csv')))
        if not files: raise FileNotFoundError(f"no CSV motions found in {path}")
        names=[]; ps=[]; Rs=[]; qs=[]
        for filename in files:
            raw=np.asarray(np.loadtxt(filename,delimiter=','),dtype=np.float32); p=jnp.asarray(raw[:,:3]); q=jnp.asarray(raw[:,7:36]); xyzw=jnp.asarray(raw[:,3:7]); quat=jnp.concatenate((xyzw[:,3:4],xyzw[:,:3]),-1)
            n=int(np.floor((raw.shape[0]-1)/src_fps/CONTROL_DT))+1; t=jnp.arange(n,dtype=jnp.float32)*CONTROL_DT*src_fps; i0=jnp.minimum(jnp.floor(t).astype(jnp.int32),raw.shape[0]-2); a=jnp.clip(t-i0.astype(jnp.float32),0,1)
            p=p[i0]*(1-a[:,None])+p[i0+1]*a[:,None]; q=q[i0]*(1-a[:,None])+q[i0+1]*a[:,None]; quat=_quat_slerp(quat[i0],quat[i0+1],a)
            names.append(Path(filename).stem); ps.append(p); Rs.append(quat_to_mat(quat)); qs.append(q)
        obj=cls.__new__(cls); _assemble(obj,names,ps,Rs,qs,kin); return obj

    def future_frames(self,f,K):
        end=self.clip_end[self.clip_id[f]]; return jnp.minimum(f[...,None]+jnp.arange(1,K+1),end[...,None])

    def goal(self,f,K):
        fut=self.future_frames(f,K); requested=f[...,None]+jnp.arange(1,K+1); recorded=requested<=self.clip_end[self.clip_id[f]][...,None]
        v=jnp.where(recorded[...,None],self.root_v_local[fut],0); w=jnp.where(recorded[...,None],self.root_w_local[fut],0); qd=jnp.where(recorded[...,None],self.qd[fut],0)
        H=yaw_mat(self.root_R[f]); Ht=jnp.swapaxes(H,-1,-2)[:,None]; dp=jnp.einsum('bkij,bkj->bki',Ht,self.root_p[fut]-self.root_p[f,None]); rot=mat_to_6d(jnp.einsum('bkij,bkjl->bkil',Ht,self.root_R[fut]))
        return jnp.concatenate((dp,rot,v,w,self.q[fut],qd*.1),-1).reshape((f.shape[0],-1))
