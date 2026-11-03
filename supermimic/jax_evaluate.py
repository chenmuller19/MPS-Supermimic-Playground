"""Deterministic per-clip evaluator for native JAX SuperMimic checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from .jax_checkpoint import restore
from .jax_geometry import JAXG1Kinematics
from .jax_mdp import JAXMDP, MAX_STEP_REWARD
from .jax_mjx_backend import JAXMJXG1
from .jax_motion import JAXMotion
from .jax_networks import NormState, policy_apply
from .jax_env import JAXTrackingEnv
from .robot import default_joint_pos_list


def evaluate(checkpoint, backend='mps', xml=None, motion_dir=None):
    ck=restore(checkpoint); cfg=ck['config']; root=Path(__file__).resolve().parents[1]
    for group,names in ((ck['policy'],('obs_norm','goal_norm')),(ck['critic'],('norm',)),(ck['wm'],('in_norm','out_norm'))):
        for name in names:
            value=group[name]
            if isinstance(value,dict): group[name]=NormState(value['mean'],value['var'],value['count'])
    xml=Path(xml or cfg.get('xml') or root/'data/menagerie_unitree_g1/scene_mjx.xml')
    motion_dir=Path(motion_dir or cfg.get('motion_dir') or root/'data/origin_lafan1_g1')
    kin=JAXG1Kinematics(str(xml)); motion=JAXMotion.from_csv_dir(motion_dir,kin)
    count=motion.num_clips; sim=JAXMJXG1(xml,count,backend); mdp=JAXMDP(kin,motion,jnp.asarray(default_joint_pos_list(),jnp.float32))
    env=JAXTrackingEnv(sim,motion,kin,mdp,gamma=cfg['gamma'],seed=cfg['seed'],eval_mode=True)
    frames=motion.clip_start+jnp.zeros_like(motion.clip_start); env.reset(jnp.ones((count,),jnp.bool_),frames)
    hist=jnp.zeros((count,cfg['hist'],99),jnp.float32); ret=jnp.zeros((count,),jnp.float32); steps=jnp.zeros((count,),jnp.int32); first=True
    clip_end=motion.clip_end
    while not bool(jnp.all(env.done_eval)):
        S=__import__('supermimic.jax_native_train',fromlist=['state_from_data']).state_from_data(sim.data)
        obs=mdp.obs(S,env.state.a_prev,env.state.xi,env.state.frame)
        hist=jnp.broadcast_to(obs[:,None,:],hist.shape) if first else jnp.concatenate((hist[:,1:],obs[:,None,:]),axis=1); first=False
        goal=motion.goal(env.state.frame,cfg['goal_frames']); action=policy_apply(ck['policy'],hist,goal)
        alive=~env.done_eval; tr=env.step(action); ret=ret+jnp.where(alive,tr['r'],0); steps=steps+alive.astype(jnp.int32)
        # Keep completed clips in range while other clips finish, matching origin evaluate.py.
        env.state=env.state._replace(frame=jnp.minimum(env.state.frame,clip_end-1))
    normalized=ret/(jnp.maximum(motion.clip_len-1,1)*MAX_STEP_REWARD)
    return {'norm_return':float(jnp.mean(normalized)),'norm_return_framew':float(jnp.sum(ret)/(jnp.sum(motion.clip_len-1)*MAX_STEP_REWARD)),'n_failed':int(jnp.sum(env.failed_eval)),'per_clip':{name:float(normalized[i]) for i,name in enumerate(motion.names)},'steps':steps.tolist()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--backend',default='mps',choices=('mps','cpu')); p.add_argument('--xml'); p.add_argument('--motion-dir'); a=p.parse_args(); print(evaluate(a.checkpoint,a.backend,a.xml,a.motion_dir))

if __name__=='__main__': main()
