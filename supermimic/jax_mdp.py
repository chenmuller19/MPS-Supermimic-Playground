"""Torch-free SuperMimic observation, reward, and termination heads."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_geometry import mat_to_6d, rot_angle_sq, yaw_mat
from .jax_motion import CONTROL_DT, JAXMotion

STATE_DIM=71; OBS_DIM=99; XI_DIM=33; NUM_J=29
MAX_STEP_REWARD=(4.0+1.0)*CONTROL_DT


def flatten_state(S):
    p,R,v,w,q,qd=S
    from .jax_geometry import mat_to_quat
    return jnp.concatenate((p,mat_to_quat(R),v,w,q,qd),-1)


def unflatten_state(x):
    from .jax_geometry import quat_to_mat
    return x[...,0:3],quat_to_mat(x[...,3:7]),x[...,7:10],x[...,10:13],x[...,13:42],x[...,42:71]


def invariant_features(S):
    p,R,v,w,q,qd=S; H=yaw_mat(R); Rt=jnp.swapaxes(R,-1,-2)
    return jnp.concatenate((p[...,2:3],mat_to_6d(jnp.einsum('...ij,...jk->...ik',jnp.swapaxes(H,-1,-2),R)),jnp.einsum('...ij,...j->...i',Rt,v),jnp.einsum('...ij,...j->...i',Rt,w),q,qd),-1)


def normalize_xi(xi):
    return jnp.concatenate(((xi[...,:1]-.75)/.45,xi[...,1:30]/.01,xi[...,30:33]/.05),-1)


class JAXMDP:
    def __init__(self, kin, motion: JAXMotion, default_q, term_height=.25, term_tilt=.8):
        self.kin=kin; self.motion=motion; self.default_q=default_q; self.soft_lo=(kin.q_lo+kin.q_hi)/2-(kin.q_hi-kin.q_lo)*.45; self.soft_hi=(kin.q_lo+kin.q_hi)/2+(kin.q_hi-kin.q_lo)*.45; self.term_height=term_height; self.term_tilt=term_tilt

    def obs(self,S,a_prev,xi,f):
        _,R,_,w,q,qd=S; Rt=jnp.swapaxes(R,-1,-2); rel=mat_to_6d(jnp.einsum('...ij,...jk->...ik',Rt,self.motion.root_R[f])); g_b=-R[...,2,:]
        return jnp.concatenate((q-self.default_q-xi[...,1:30],qd,jnp.einsum('...ij,...j->...i',Rt,w),g_b,a_prev,rel),-1)

    def add_noise(self,o,key):
        scale=jnp.concatenate((jnp.full((29,),.01),jnp.full((29,),.5),jnp.full((3,),.2),jnp.full((3,),.05),jnp.zeros((29,)),jnp.full((6,),.05)))
        return o+(2*jax_random_uniform(key,o.shape)-1)*scale

    def critic_features(self,S,f,xi,body_positions=None):
        p,R,v,w,q,qd=S; H=yaw_mat(R); Ht=jnp.swapaxes(H,-1,-2); rp,rR=self.motion.root_p[f],self.motion.root_R[f]
        bp=self.kin.forward(p,R,v,w,q,qd)[0] if body_positions is None else body_positions; dR=yaw_mat(jnp.einsum('...ij,...kj->...ik',R,rR)); anchor=jnp.concatenate((p[...,:2],rp[...,2:3]),-1)
        ref_bp=anchor[...,None,:]+jnp.einsum('...ij,...kj->...ki',dR,self.motion.body_p[f]-rp[...,None,:]); e_body=jnp.einsum('...ij,...kj->...ki',Ht,ref_bp-bp).reshape(p.shape[:-1]+(-1,))
        e_root=jnp.einsum('...ij,...j->...i',Ht,rp-p); rel=mat_to_6d(jnp.einsum('...ij,...jk->...ik',jnp.swapaxes(R,-1,-2),rR))
        return jnp.concatenate((invariant_features(S),self.motion.q[f]-q,e_body,e_root,rel,xi),-1)

    def reward(self,S,f,a,a_prev,body_kinematics=None):
        p,R,v,w,q,qd=S
        bp,bR,bv,bw=self.kin.forward(p,R,v,w,q,qd) if body_kinematics is None else body_kinematics
        rp,rR=self.motion.root_p[f],self.motion.root_R[f]
        dR=yaw_mat(jnp.einsum('...ij,...kj->...ik',R,rR)); anchor=jnp.concatenate((p[...,:2],rp[...,2:3]),-1)
        ref_bp=anchor[...,None,:]+jnp.einsum('...ij,...kj->...ki',dR,self.motion.body_p[f]-rp[...,None,:]); ref_bR=jnp.einsum('...ij,...kjl->...kil',dR,self.motion.body_R[f])
        e_pos=jnp.mean(jnp.sum((ref_bp-bp)**2,-1),-1); e_ori=jnp.mean(rot_angle_sq(ref_bR,bR),-1); e_lv=jnp.mean(jnp.sum((self.motion.body_v[f]-bv)**2,-1),-1); e_av=jnp.mean(jnp.sum((self.motion.body_w[f]-bw)**2,-1),-1)
        e_rp=jnp.sum((rp-p)**2,-1); e_ro=rot_angle_sq(rR,R)
        track=jnp.exp(-e_pos/.3**2)+jnp.exp(-e_ori/.4**2)+jnp.exp(-e_lv/1.0)+jnp.exp(-e_av/3.14**2)+.5*jnp.exp(-e_rp/.3**2)+.5*jnp.exp(-e_ro/.4**2)
        rate=jnp.sum((a-a_prev)**2,-1); lim=jnp.sum(jnp.maximum(q-self.soft_hi,0)+jnp.maximum(self.soft_lo-q,0),-1)
        return (track-.1*rate-2*lim)*CONTROL_DT

    def termination(self,S,f):
        p,R=S[0],S[1]; rp,rR=self.motion.root_p[f],self.motion.root_R[f]
        return (jnp.abs(rp[...,2]-p[...,2])>self.term_height)|(jnp.abs(rR[...,2,2]-R[...,2,2])>self.term_tilt)

    def soft_termination(self,S,f,tz=.02,tt=.04):
        p,R=S[0],S[1]; rp,rR=self.motion.root_p[f],self.motion.root_R[f]; ez=jnp.abs(rp[...,2]-p[...,2]); et=jnp.abs(rR[...,2,2]-R[...,2,2]); alive=(1-jax_sigmoid((ez-self.term_height)/tz))*(1-jax_sigmoid((et-self.term_tilt)/tt)); return 1-alive


def jax_sigmoid(x): return 1/(1+jnp.exp(-x))

def jax_random_uniform(key,shape): return jax.random.uniform(key,shape,dtype=jnp.float32)
