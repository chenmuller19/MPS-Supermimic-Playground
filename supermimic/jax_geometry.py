"""JAX-native rotations and G1 analytic forward kinematics."""

from __future__ import annotations

import mujoco
import jax
import jax.numpy as jnp

from .robot import G1_JOINT_NAMES, TRACKED_BODIES


def quat_to_mat(q):
    q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    return jnp.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def mat_to_quat(m):
    m00,m01,m02=m[...,0,0],m[...,0,1],m[...,0,2]
    m10,m11,m12=m[...,1,0],m[...,1,1],m[...,1,2]
    m20,m21,m22=m[...,2,0],m[...,2,1],m[...,2,2]
    q0=jnp.sqrt(jnp.maximum(1+m00+m11+m22,1e-10))/2
    q1=jnp.sqrt(jnp.maximum(1+m00-m11-m22,1e-10))/2
    q2=jnp.sqrt(jnp.maximum(1-m00+m11-m22,1e-10))/2
    q3=jnp.sqrt(jnp.maximum(1-m00-m11+m22,1e-10))/2
    cand=jnp.stack((
        jnp.stack((q0,(m21-m12)/(4*jnp.maximum(q0,1e-6)),(m02-m20)/(4*jnp.maximum(q0,1e-6)),(m10-m01)/(4*jnp.maximum(q0,1e-6))),-1),
        jnp.stack(((m21-m12)/(4*jnp.maximum(q1,1e-6)),q1,(m01+m10)/(4*jnp.maximum(q1,1e-6)),(m02+m20)/(4*jnp.maximum(q1,1e-6))),-1),
        jnp.stack(((m02-m20)/(4*jnp.maximum(q2,1e-6)),(m01+m10)/(4*jnp.maximum(q2,1e-6)),q2,(m12+m21)/(4*jnp.maximum(q2,1e-6))),-1),
        jnp.stack(((m10-m01)/(4*jnp.maximum(q3,1e-6)),(m02+m20)/(4*jnp.maximum(q3,1e-6)),(m12+m21)/(4*jnp.maximum(q3,1e-6)),q3),-1)), -2)
    idx=jnp.argmax(jnp.stack((q0,q1,q2,q3),-1),-1)
    q=jnp.take_along_axis(cand,idx[...,None,None],axis=-2)[...,0,:]
    q=q/jnp.maximum(jnp.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    return q*jnp.where(q[...,:1]<0,-1.0,1.0)


def skew(v):
    x,y,z=jnp.moveaxis(v,-1,0); o=jnp.zeros_like(x)
    return jnp.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(v.shape[:-1]+(3,3))


def so3_exp(v):
    theta2=jnp.sum(v*v,-1,keepdims=True); theta=jnp.sqrt(theta2+1e-12)
    a=jnp.where(theta2<1e-8,1-theta2/6,jnp.sin(theta)/theta)
    b=jnp.where(theta2<1e-8,.5-theta2/24,(1-jnp.cos(theta))/(theta2+1e-12))
    k=skew(v); eye=jnp.broadcast_to(jnp.eye(3,dtype=v.dtype),k.shape)
    return eye+a[...,None]*k+b[...,None]*(k@k)


def quat_to_rotvec(q):
    q=q*jnp.where(q[...,:1]<0,-1.0,1.0); v=q[...,1:]
    norm=jnp.linalg.norm(v,axis=-1,keepdims=True)
    half=jnp.arctan2(norm,q[...,:1])
    scale=jnp.where(norm<1e-6,2/jnp.maximum(q[...,:1],1e-6),2*half/(norm+1e-12))
    return v*scale


def so3_log(r):
    return quat_to_rotvec(mat_to_quat(r))


def yaw_mat(r):
    fx,fy=r[...,0,0],r[...,1,0]; n=jnp.sqrt(fx*fx+fy*fy+1e-9)
    c,s=fx/n,fy/n; o=jnp.zeros_like(c); one=jnp.ones_like(c)
    return jnp.stack((c,-s,o,s,c,o,o,o,one),-1).reshape(r.shape[:-2]+(3,3))


def mat_to_6d(r):
    return r[...,:,:2].reshape(r.shape[:-2]+(6,))


def rot_angle_sq(ra,rb):
    trace=jnp.sum(ra*rb,axis=(-1,-2))
    cosine=jnp.clip((trace-1)/2,-1+1e-6,1-1e-6)
    return jnp.arccos(cosine)**2


class JAXG1Kinematics:
    """Differentiable analytic FK with MJCF-derived static constants."""

    def __init__(self, xml_path: str):
        model=mujoco.MjModel.from_xml_path(xml_path)
        names_to_idx={name:i for i,name in enumerate(G1_JOINT_NAMES)}
        body_names=[]; parent=[]; joint_idx=[]; axes=[]; off_p=[]; off_r=[]
        mj_to_local={}
        for b in range(1,model.nbody):
            name=model.body(b).name
            if model.body_jntnum[b]==0: continue
            jid=model.body_jntadr[b]; mj_to_local[b]=len(body_names); body_names.append(name)
            if model.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1); joint_idx.append(-1)
            else:
                pb=model.body_parentid[b]
                if pb not in mj_to_local: raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb]); joint_idx.append(names_to_idx[model.joint(jid).name])
            axes.append(model.jnt_axis[jid].copy()); off_p.append(model.body_pos[b].copy()); off_r.append(model.body_quat[b].copy())
        self.body_names=tuple(body_names)
        self.parent=tuple(parent); self.joint_idx=tuple(joint_idx)
        self.axes=jnp.asarray(axes,dtype=jnp.float32); self.off_p=jnp.asarray(off_p,dtype=jnp.float32)
        q=jnp.asarray(off_r,dtype=jnp.float32); self.off_r=quat_to_mat(q)
        self.tracked_idx=tuple(body_names.index(name) for name in TRACKED_BODIES)
        depths=[]
        for par in self.parent:
            depths.append(0 if par<0 else depths[par]+1)
        groups=[]
        for level in range(1,max(depths,default=0)+1):
            ids=tuple(i for i,d in enumerate(depths) if d==level and self.parent[i]>=0)
            if ids:
                groups.append((ids,tuple(self.parent[i] for i in ids),tuple(self.joint_idx[i] for i in ids)))
        self.forward_groups=tuple(groups)
        lo=jnp.zeros((29,),jnp.float32); hi=jnp.zeros((29,),jnp.float32)
        for jid in range(model.njnt):
            name=model.joint(jid).name
            if name in names_to_idx:
                i=names_to_idx[name]; lo=lo.at[i].set(model.jnt_range[jid,0]); hi=hi.at[i].set(model.jnt_range[jid,1])
        self.q_lo,self.q_hi=lo,hi

    def forward(self,p,r,v,w,q,qd,all_bodies=False):
        n=len(self.body_names)
        positions=[None]*n; rotations=[None]*n; velocities=[None]*n; angular=[None]*n
        for i,par in enumerate(self.parent):
            if par<0:
                positions[i],rotations[i],velocities[i],angular[i]=p,r,v,w
        for ids,parents,joints in self.forward_groups:
            rp=jnp.stack([rotations[k] for k in parents],axis=-3)
            pp=jnp.stack([positions[k] for k in parents],axis=-2)
            vp=jnp.stack([velocities[k] for k in parents],axis=-2)
            wp=jnp.stack([angular[k] for k in parents],axis=-2)
            body_ids=jnp.asarray(ids,dtype=jnp.int32)
            joint_ids=jnp.asarray(joints,dtype=jnp.int32)
            off_p=self.off_p[body_ids]; off_r=self.off_r[body_ids]; axes=self.axes[body_ids]
            qg=q[...,joint_ids]; qdg=qd[...,joint_ids]
            pi=pp+jnp.einsum('...gij,gj->...gi',rp,off_p)
            ro=jnp.einsum('...gij,gjk->...gik',rp,off_r)
            ri=ro@so3_exp(axes*qg[...,None])
            axis=jnp.einsum('...gij,gj->...gi',ro,axes)
            wi=wp+axis*qdg[...,None]
            vi=vp+jnp.cross(wp,pi-pp)
            for k,i in enumerate(ids):
                positions[i],rotations[i]=pi[...,k,:],ri[...,k,:,:]
                velocities[i],angular[i]=vi[...,k,:],wi[...,k,:]
        idx=range(n) if all_bodies else self.tracked_idx
        return tuple(jnp.stack([arr[k] for k in idx],axis=-2 if a!=1 else -3) for a,arr in enumerate((positions,rotations,velocities,angular)))
