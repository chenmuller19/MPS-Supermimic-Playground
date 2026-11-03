"""JAX episode/reset bookkeeping matching origin TrackingSim semantics."""

from __future__ import annotations
from typing import NamedTuple
import time
import jax
import jax.numpy as jnp
from .jax_geometry import mat_to_quat, quat_to_mat, so3_exp
from .jax_mdp import MAX_STEP_REWARD, flatten_state


def _sync_tree(value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf,'block_until_ready'): leaf.block_until_ready()


class EnvState(NamedTuple):
    frame: jax.Array
    ep_step: jax.Array
    a_prev: jax.Array
    start_bin: jax.Array
    ep_disc_ret: jax.Array
    ep_ret: jax.Array
    bin_G: jax.Array
    bin_w_avg: jax.Array
    xi: jax.Array


def _update_bin_returns(bin_g, start_bin, done, returns):
    n_bins=bin_g.shape[0]
    indices=jnp.where(done,start_bin,n_bins)
    padded=jnp.concatenate((bin_g,jnp.zeros((1,),dtype=bin_g.dtype)))
    updates=jnp.where(done,.8*bin_g[start_bin]+.2*returns,0.)
    return padded.at[indices].set(updates)[:n_bins]


class JAXTrackingEnv:
    def __init__(self,sim,motion,kin,mdp,gamma=.99,max_episode_steps=500,seed=0,eval_mode=False):
        self.sim,self.motion,self.kin,self.mdp=sim,motion,kin,mdp
        self.profile=False
        self.profile_parts={'physics_s':0.0,'mdp_s':0.0,'bookkeeping_s':0.0,'reset_s':0.0,'count':0}
        self.phys_state=None
        self.gamma,self.max_episode_steps,self.eval_mode=gamma,(10**9 if eval_mode else max_episode_steps),eval_mode
        self.done_eval=jnp.zeros((sim.num_envs,),jnp.bool_); self.failed_eval=jnp.zeros((sim.num_envs,),jnp.bool_)
        self.G_max=MAX_STEP_REWARD/(1-gamma); self.key=jax.random.key(seed)
        self.state=EnvState(jnp.zeros((sim.num_envs,),jnp.int32),jnp.zeros((sim.num_envs,),jnp.int32),jnp.zeros((sim.num_envs,29),jnp.float32),jnp.zeros((sim.num_envs,),jnp.int32),jnp.zeros((sim.num_envs,),jnp.float32),jnp.zeros((sim.num_envs,),jnp.float32),jnp.zeros((motion.num_bins,),jnp.float32),jnp.full((motion.num_bins,),1/motion.num_bins,jnp.float32),jnp.zeros((sim.num_envs,33),jnp.float32))
        if eval_mode: xi=self.state.xi.at[:,0].set(1.0)
        else:
            kf,kcom=jax.random.split(self._next_key())
            fr=jax.random.uniform(kf,(sim.num_envs,),minval=.3,maxval=1.2)
            com=jax.random.uniform(kcom,(sim.num_envs,3),minval=jnp.array([-.025,-.05,-.05]),maxval=jnp.array([.025,.05,.05]))
            xi=self.state.xi.at[:,0].set(fr).at[:,30:33].set(com)
        self.state=self.state._replace(xi=xi)
        self.reset(jnp.ones((sim.num_envs,),jnp.bool_))

    def _next_key(self):
        self.key,k=jax.random.split(self.key); return k

    def sample_frames(self,n):
        soft=jax.nn.softmax(-25*self.state.bin_G/self.G_max)
        avg=.999*self.state.bin_w_avg+.001*soft
        probs=.5/self.motion.num_bins+.4*soft+.1*avg/jnp.sum(avg)
        self.state=self.state._replace(bin_w_avg=avg)
        kb,kf=self._next_key(),self._next_key()
        bins=jax.random.choice(kb,self.motion.num_bins,(n,),replace=True,p=probs)
        start=self.motion.bin_start[bins]; end=jnp.minimum(start+50,self.motion.clip_end[self.motion.bin_clip[bins]])
        frame=start+jnp.floor(jax.random.uniform(kf,(n,))*(end-start)).astype(jnp.int32)
        return frame,bins

    def _reference_state(self,frames,noise=True):
        p,R,v,w,q,qd=(self.motion.root_p[frames],self.motion.root_R[frames],self.motion.root_v[frames],self.motion.root_w[frames],self.motion.q[frames],self.motion.qd[frames])
        if noise and not self.eval_mode:
            keys=jax.random.split(self._next_key(),5)
            def uniform(k,lo,hi,shape): return jax.random.uniform(k,shape,minval=lo,maxval=hi)
            p=p+uniform(keys[0],jnp.array([-.05,-.05,-.01]),jnp.array([.05,.05,.01]),p.shape)
            rv=uniform(keys[1],jnp.array([-.1,-.1,-.2]),jnp.array([.1,.1,.2]),p.shape); R=so3_exp(rv)@R
            v=v+uniform(keys[2],jnp.array([-.5,-.5,-.2]),jnp.array([.5,.5,.2]),v.shape)
            w=w+uniform(keys[3],jnp.array([-.52,-.52,-.78]),jnp.array([.52,.52,.78]),w.shape)
            q=q+uniform(keys[4],-.1,.1,q.shape)
        q=jnp.clip(q,self.kin.q_lo,self.kin.q_hi)
        return p,R,v,w,q,qd

    def reset(self,mask,frames=None):
        mask=jnp.asarray(mask,jnp.bool_); n=self.sim.num_envs
        avg_before=self.state.bin_w_avg
        if frames is None:
            frame,bins=self.sample_frames(n); avg_sampled=self.state.bin_w_avg
        else:
            frame=jnp.asarray(frames,jnp.int32); bins=self.motion.frame_bin[frame]; avg_sampled=avg_before
        old=self.state
        frame=jnp.where(mask,frame,old.frame); bins=jnp.where(mask,bins,old.start_bin)
        p,R,v,w,q,qd=self._reference_state(frame,noise=True)
        qpos=jnp.concatenate((p,mat_to_quat(R),q),-1); qvel=jnp.concatenate((v,w,qd),-1)
        xi=old.xi
        if not self.eval_mode:
            ko=self._next_key(); off=jax.random.uniform(ko,(n,29),minval=-.01,maxval=.01)
            xi=xi.at[:,1:30].set(jnp.where(mask[:,None],off,xi[:,1:30]))
        avg=jnp.where(jnp.any(mask),avg_sampled,avg_before)
        self.state=EnvState(frame,jnp.where(mask,0,old.ep_step),jnp.where(mask[:,None],0,old.a_prev),bins,jnp.where(mask,0,old.ep_disc_ret),jnp.where(mask,0,old.ep_ret),old.bin_G,avg,xi)
        self.sim.set_xi_jax(xi)
        self.sim.reset_masked_jax(mask,qpos,qvel)
        data=self.sim.data
        self.phys_state=(data.qpos[:,:3],quat_to_mat(data.qpos[:,3:7]),data.qvel[:,:3],data.qvel[:,3:6],data.qpos[:,7:36],data.qvel[:,6:35])
        return data

    def step_pure(self, action, carry):
        """JAX-only environment transition suitable for jit and lax.scan."""
        old,key,model,data,phys=carry
        data1=self.sim.step_data(model,data,action,joint_offset=old.xi[:,1:30])
        S1=(data1.qpos[:,:3],quat_to_mat(data1.qpos[:,3:7]),data1.qvel[:,:3],data1.qvel[:,3:6],data1.qpos[:,7:36],data1.qvel[:,6:35])
        frame1=old.frame+1
        end=self.motion.clip_end[self.motion.clip_id[old.frame]]
        reward=self.mdp.reward(S1,frame1,action,old.a_prev)
        term=self.mdp.termination(S1,frame1)
        done=term|(frame1>=end)|(old.ep_step+1>=self.max_episode_steps)
        transition={'s':flatten_state(phys),'a':action,'a_prev':old.a_prev,'f':old.frame,'s1':flatten_state(S1),'r':reward,'term':term,'done':done,'ep_step':old.ep_step,'xi':old.xi,'clip_end':end}
        ep_disc=old.ep_disc_ret+(self.gamma**old.ep_step.astype(jnp.float32))*reward
        ep_ret=old.ep_ret+reward; ep_len=old.ep_step+1
        boot=jnp.where(term,0,(self.gamma**ep_len.astype(jnp.float32))*(ep_ret/ep_len)/(1-self.gamma))
        G=ep_disc+boot
        bg=old.bin_G if self.eval_mode else _update_bin_returns(old.bin_G,old.start_bin,done,G)
        stepped=old._replace(frame=frame1,ep_step=ep_len,a_prev=action,ep_disc_ret=ep_disc,ep_ret=ep_ret,bin_G=bg)

        def no_reset(args):
            state,key,model,data,phys=args
            soft=jax.nn.softmax(-25*state.bin_G/self.G_max)
            state=state._replace(bin_w_avg=.999*state.bin_w_avg+.001*soft)
            for _ in range(4): key,_=jax.random.split(key)
            return state,key,model,data,phys

        def reset_done(args):
            state,key,model,data,phys=args
            avg_before=state.bin_w_avg
            soft=jax.nn.softmax(-25*state.bin_G/self.G_max)
            avg=.999*avg_before+.001*soft
            probs=.5/self.motion.num_bins+.4*soft+.1*avg/jnp.sum(avg)
            key,kb=jax.random.split(key); key,kf=jax.random.split(key)
            bins=jax.random.choice(kb,self.motion.num_bins,(self.sim.num_envs,),replace=True,p=probs)
            start=self.motion.bin_start[bins]
            end_frame=jnp.minimum(start+50,self.motion.clip_end[self.motion.bin_clip[bins]])
            frames=start+jnp.floor(jax.random.uniform(kf,(self.sim.num_envs,))*(end_frame-start)).astype(jnp.int32)
            frames=jnp.where(done,frames,state.frame); bins=jnp.where(done,bins,state.start_bin)
            p,R,v,w,q,qd=(self.motion.root_p[frames],self.motion.root_R[frames],self.motion.root_v[frames],self.motion.root_w[frames],self.motion.q[frames],self.motion.qd[frames])
            key,kref=jax.random.split(key); keys=jax.random.split(kref,5)
            def uniform(k,lo,hi,shape): return jax.random.uniform(k,shape,minval=lo,maxval=hi)
            p=p+uniform(keys[0],jnp.array([-.05,-.05,-.01]),jnp.array([.05,.05,.01]),p.shape)
            rv=uniform(keys[1],jnp.array([-.1,-.1,-.2]),jnp.array([.1,.1,.2]),p.shape); R=so3_exp(rv)@R
            v=v+uniform(keys[2],jnp.array([-.5,-.5,-.2]),jnp.array([.5,.5,.2]),v.shape)
            w=w+uniform(keys[3],jnp.array([-.52,-.52,-.78]),jnp.array([.52,.52,.78]),w.shape)
            q=jnp.clip(q+uniform(keys[4],-.1,.1,q.shape),self.kin.q_lo,self.kin.q_hi)
            qpos=jnp.concatenate((p,mat_to_quat(R),q),-1); qvel=jnp.concatenate((v,w,qd),-1)
            key,ko=jax.random.split(key)
            off=jax.random.uniform(ko,(self.sim.num_envs,29),minval=-.01,maxval=.01)
            xi=state.xi.at[:,1:30].set(jnp.where(done[:,None],off,state.xi[:,1:30]))
            state=EnvState(frames,jnp.where(done,0,state.ep_step),jnp.where(done[:,None],0,state.a_prev),bins,jnp.where(done,0,state.ep_disc_ret),jnp.where(done,0,state.ep_ret),state.bin_G,avg,xi)
            model=self.sim.model_batch_from_xi(xi)
            data=self.sim.reset_masked_data(model,data,done,qpos,qvel)
            phys=(data.qpos[:,:3],quat_to_mat(data.qpos[:,3:7]),data.qvel[:,:3],data.qvel[:,3:6],data.qpos[:,7:36],data.qvel[:,6:35])
            return state,key,model,data,phys

        if self.eval_mode:
            result=(stepped,key,model,data1,S1)
        else:
            result=jax.lax.cond(jnp.any(done),reset_done,no_reset,(stepped,key,model,data1,S1))
        return result,transition

    def step(self,action):
        t0=time.perf_counter() if self.profile else 0.0
        old=self.state; Sdata=self.sim.data
        from .jax_native_train import state_from_data
        S=self.phys_state; frame0=old.frame
        data=self.sim.step_jax(action,joint_offset=old.xi[:,1:30]); S1=state_from_data(data); self.phys_state=S1; frame1=frame0+1
        if self.profile: _sync_tree(data.qpos)
        t1=time.perf_counter() if self.profile else 0.0
        end=self.motion.clip_end[self.motion.clip_id[frame0]]
        reward,terms=self.mdp.reward(S1,frame1,action,old.a_prev),None
        term=self.mdp.termination(S1,frame1); done=term|(frame1>=end)|(old.ep_step+1>=self.max_episode_steps)
        transition={'s':flatten_state(S),'a':action,'a_prev':old.a_prev,'f':frame0,'s1':flatten_state(S1),'r':reward,'term':term,'done':done,'ep_step':old.ep_step,'xi':old.xi,'clip_end':end}
        if self.profile: _sync_tree((reward,term,transition['s1']))
        t2=time.perf_counter() if self.profile else 0.0
        ep_disc=old.ep_disc_ret+(self.gamma**old.ep_step.astype(jnp.float32))*reward; ep_ret=old.ep_ret+reward; ep_len=old.ep_step+1
        if self.eval_mode:
            self.failed_eval=self.failed_eval|(term&~self.done_eval); self.done_eval=self.done_eval|done
        boot=jnp.where(term,0,(self.gamma**ep_len.astype(jnp.float32))*(ep_ret/ep_len)/(1-self.gamma)); G=ep_disc+boot
        bg=old.bin_G if self.eval_mode else _update_bin_returns(old.bin_G,old.start_bin,done,G)
        self.state=old._replace(frame=frame1,ep_step=ep_len,a_prev=action,ep_disc_ret=ep_disc,ep_ret=ep_ret,bin_G=bg)
        if self.profile: _sync_tree(bg)
        t3=time.perf_counter() if self.profile else 0.0
        if not self.eval_mode:
            if bool(jax.device_get(jnp.any(done))):
                self.reset(done)
            else:
                soft=jax.nn.softmax(-25*self.state.bin_G/self.G_max)
                avg=.999*self.state.bin_w_avg+.001*soft
                self.state=self.state._replace(bin_w_avg=avg)
                for _ in range(4): self._next_key()
        if self.profile:
            _sync_tree(self.sim.data.qpos)
            t4=time.perf_counter()
            self.profile_parts['physics_s']+=t1-t0; self.profile_parts['mdp_s']+=t2-t1
            self.profile_parts['bookkeeping_s']+=t3-t2; self.profile_parts['reset_s']+=t4-t3
            self.profile_parts['count']+=1
        return transition
