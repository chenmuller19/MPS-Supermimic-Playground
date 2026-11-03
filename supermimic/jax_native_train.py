"""Torch-free JAX/MJX training smoke for the SuperMimic learner.

This is the native path: MJX collection, replay, world-model gradients,
imagined rollout, policy gradients, and critic GAE updates stay on the JAX device.
The existing Torch trainer remains as a separate parity baseline.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from .jax_buffer import JAXReplay, sample_starts, sample_windows
from .jax_env import JAXTrackingEnv
from .jax_geometry import JAXG1Kinematics, mat_to_quat, so3_exp, so3_log, yaw_mat
from .jax_mdp import JAXMDP, flatten_state, invariant_features, normalize_xi, unflatten_state
from .jax_mjx_backend import JAXMJXG1
from .jax_motion import JAXMotion
from .jax_networks import critic_apply, init_critic, init_policy, init_world_model, policy_apply, update_norm, world_model_apply
from .robot import default_joint_pos_list


def state_from_data(data):
    from .jax_geometry import quat_to_mat
    qpos, qvel = data.qpos, data.qvel
    return qpos[:, :3], quat_to_mat(qpos[:, 3:7]), qvel[:, :3], qvel[:, 3:6], qpos[:, 7:36], qvel[:, 6:35]


def wm_target(S, S1):
    p,R,v,w,q,qd=S; p1,R1,v1,w1,q1,qd1=S1; H=yaw_mat(R); Rt=jnp.swapaxes(R,-1,-2)
    return jnp.concatenate((
        jnp.einsum('...ij,...j->...i',jnp.swapaxes(H,-1,-2),p1-p),
        so3_log(jnp.einsum('...ij,...jk->...ik',Rt,R1)),
        jnp.einsum('...ij,...j->...i',Rt,v1-v), jnp.einsum('...ij,...j->...i',Rt,w1-w), q1-q, qd1-qd),-1)


def wm_state_error(pred, ref):
    p,R,v,w,q,qd=pred; ph,Rh,vh,wh,qh,qdh=ref; H=yaw_mat(R); Rt=jnp.swapaxes(R,-1,-2)
    return jnp.concatenate((jnp.einsum('...ij,...j->...i',jnp.swapaxes(H,-1,-2),ph-p),so3_log(Rt@Rh),jnp.einsum('...ij,...j->...i',Rt,vh-v),jnp.einsum('...ij,...j->...i',Rt,wh-w),qh-q,qdh-qd),-1)


def wm_component_mse(error, std, error_floor):
    normalized=(error/jnp.maximum(std,error_floor))**2
    return jnp.mean(normalized,axis=-1)


def wm_step(params, S, action, xi):
    p,R,v,w,q,qd=S; y=world_model_apply(params,jnp.concatenate((invariant_features(S),action,normalize_xi(xi)),-1)); dth=y[...,3:6]; H=yaw_mat(R)
    return (p+jnp.einsum('...ij,...j->...i',H,y[...,:3]),R@so3_exp(dth),v+jnp.einsum('...ij,...j->...i',R,y[...,6:9]),w+jnp.einsum('...ij,...j->...i',R,y[...,9:12]),q+y[...,12:41],qd+y[...,41:70])


def _sync_tree(value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf,'block_until_ready'): leaf.block_until_ready()


def _memory_stats_text(backend):
    if backend != 'mps':
        return ''
    stats = jax.devices('mps')[0].memory_stats() or {}
    def gib(name):
        value = stats.get(name)
        return f'{value / 2**30:.2f}' if value is not None else 'na'
    return f' | mem_active {gib("bytes_in_use")}GiB | mem_pool {gib("pool_bytes")}GiB | mem_peak {gib("peak_bytes_in_use")}GiB'


def compiled_world_update(wm, opt_state, batch, lr, opt, error_floor, window):
    """One fixed-shape world-model update; the caller caches this with jax.jit."""
    S0=unflatten_state(batch['s'][:,0])
    def loss_fn(net):
        params={**wm,'net':net}
        std=jnp.sqrt(wm['out_norm'].var+1e-8)
        inputs=(
            jnp.swapaxes(batch['a'],0,1),
            jnp.swapaxes(batch['xi'],0,1),
            jnp.swapaxes(batch['s1'],0,1),
            jnp.swapaxes(batch['mask'],0,1),
        )
        def loss_step(S,step):
            action,xi,target,mask=step
            S=wm_step(params,S,action,xi)
            err=wm_state_error(S,unflatten_state(target))
            component_loss=wm_component_mse(err,std,error_floor)
            return S,component_loss*mask
        _,losses=jax.lax.scan(loss_step,S0,inputs)
        return jnp.sum(losses)/batch['s'].shape[0]
    loss,grads=jax.value_and_grad(loss_fn)(wm['net'])
    updates,new_opt=opt.update(grads,opt_state,wm['net'])
    updates=jax.tree_util.tree_map(lambda x:x*lr,updates)
    return {**wm,'net':optax.apply_updates(wm['net'],updates)},new_opt,loss/window


def _policy_actor_stage(
    rollout_key, state0, xi0, xi_n, frame0, prev0, obs0, goal0,
    policy, critic, wm, opt_pi_state, lr,
    *, mdp, motion, opt_pi, horizon, pi_batch, gamma, lam, no_critic, goal_frames,
):
    def imagine(policy_params):
        state,frame,prev,hist,goal=state0,frame0,prev0,obs0,goal0
        actions=[]; prev_actions=[]; all_states=[state]; all_frames=[frame]; all_goals=[goal]
        for t in range(horizon):
            action=policy_apply(policy_params,hist,goal)
            state=wm_step(wm,state,action,xi0); frame=frame+1
            goal=motion.goal(frame,goal_frames)
            obs=mdp.add_noise(mdp.obs(state,action,xi0,frame),jax.random.fold_in(rollout_key,t+2))
            hist=jnp.concatenate((hist[:,1:],obs[:,None,:]),1)
            actions.append(action); prev_actions.append(prev)
            all_states.append(state); all_frames.append(frame); all_goals.append(goal)
            prev=action
        states=jax.tree_util.tree_map(lambda *xs:jnp.stack(xs),*all_states)
        frames=jnp.stack(all_frames); goals=jnp.stack(all_goals)
        flat_states=jax.tree_util.tree_map(
            lambda x:x[1:].reshape((-1,)+x.shape[2:]),states
        )
        flat_frames=frames[1:].reshape((-1,))
        actions=jnp.stack(actions); prev_actions=jnp.stack(prev_actions)
        body_positions=None
        reward_args=(
            flat_states,flat_frames,actions.reshape((-1,actions.shape[-1])),
            prev_actions.reshape((-1,prev_actions.shape[-1])),
        )
        if isinstance(mdp,JAXMDP):
            body_kinematics=mdp.kin.forward(*flat_states)
            rewards=mdp.reward(*reward_args,body_kinematics=body_kinematics)
            if not no_critic:
                body_positions=body_kinematics[0].reshape((horizon,pi_batch)+body_kinematics[0].shape[1:])
        else:
            rewards=mdp.reward(*reward_args)
        rewards=rewards.reshape((horizon,pi_batch))
        dones=mdp.soft_termination(flat_states,flat_frames).reshape((horizon,pi_batch))
        return rewards,dones,states,frames,goals,body_positions

    def actor_loss(pp):
        rollout = imagine({**policy, **pp})
        rewards, dones, states, frames, goals, body_positions = rollout
        alive = jnp.cumprod(1 - dones, axis=0)
        cont = jnp.concatenate((jnp.ones((1, pi_batch)), alive[:-1]), 0)
        disc = gamma ** jnp.arange(horizon, dtype=jnp.float32)[:, None]
        actor_return = jnp.sum(disc * cont * rewards, axis=0)
        if not no_critic:
            final_state=jax.tree_util.tree_map(lambda x:x[-1],states)
            final_body_positions=None if body_positions is None else body_positions[-1]
            final_critic_features=(
                mdp.critic_features(final_state,frames[-1],xi_n)
                if final_body_positions is None
                else mdp.critic_features(final_state,frames[-1],xi_n,body_positions=final_body_positions)
            )
            final_features = jnp.concatenate((final_critic_features, goals[-1]), -1)
            final_value = critic_apply(critic, final_features)
            actor_return = actor_return + (gamma ** horizon) * alive[-1] * jax.lax.stop_gradient(final_value)
        return -jnp.mean(actor_return), rollout

    pi_params = {'enc': policy['enc'], 'dec': policy['dec']}
    (imagined_loss, rollout), gpi = jax.value_and_grad(actor_loss, has_aux=True)(pi_params)
    upd, opt_pi_state = opt_pi.update(gpi, opt_pi_state, pi_params)
    upd = jax.tree_util.tree_map(lambda x: x * lr, upd)
    policy = {**policy, **optax.apply_updates(pi_params, upd)}
    return policy,opt_pi_state,imagined_loss,rollout


def _critic_data_stage(rollout,state0,xi_n,critic,*,mdp,horizon,pi_batch,gamma,lam):
    rewards,dones,states,frames,goals,body_positions=rollout
    flat_states=jax.tree_util.tree_map(
        lambda x:x.reshape((-1,)+x.shape[2:]),states
    )
    flat_frames=frames.reshape((-1,))
    flat_xi=jnp.broadcast_to(xi_n[None],(horizon+1,)+xi_n.shape).reshape((-1,xi_n.shape[-1]))
    if body_positions is None:
        features_all=mdp.critic_features(flat_states,flat_frames,flat_xi)
    else:
        initial_positions=mdp.kin.forward(*state0)[0]
        all_positions=jnp.concatenate((initial_positions[None],body_positions),axis=0)
        flat_positions=all_positions.reshape((-1,)+all_positions.shape[2:])
        features_all=mdp.critic_features(flat_states,flat_frames,flat_xi,body_positions=flat_positions)
    features_all=features_all.reshape((horizon+1,pi_batch,-1))
    xall=jnp.concatenate((features_all,goals),-1)
    values=critic_apply(critic,xall)
    gae=jnp.zeros_like(values[0]); targets=[]
    for t in range(horizon-1,-1,-1):
        delta=rewards[t]+gamma*(1-dones[t])*values[t+1]-values[t]
        gae=delta+gamma*lam*(1-dones[t])*gae
        targets.append(gae+values[t])
    targets=jnp.stack(targets[::-1])
    critic={**critic,'norm':update_norm(critic['norm'],xall.reshape((-1,xall.shape[-1])))}
    x=xall[:-1].reshape((-1,xall.shape[-1]))
    y=jax.lax.stop_gradient(targets.reshape((-1,)))
    return critic,x,y


def _critic_update_stage(key,critic,opt_v_state,x,y,lr,*,opt_v,critic_epochs,critic_minibatches):
    n=x.shape[0]; mb=max(n//critic_minibatches,1)
    critic_loss=jnp.asarray(0.,jnp.float32)
    for _ in range(critic_epochs):
        key,kperm=jax.random.split(key); perm=jax.random.permutation(kperm,n)
        for i in range(critic_minibatches):
            idx=perm[i*mb:min((i+1)*mb,n)]
            if idx.shape[0]==0:
                continue
            def vloss(net):
                return jnp.mean((critic_apply({**critic,'net':net},x[idx])-y[idx])**2)
            critic_loss,gv=jax.value_and_grad(vloss)(critic['net'])
            upd,opt_v_state=opt_v.update(gv,opt_v_state,critic['net'])
            upd=jax.tree_util.tree_map(lambda z:z*lr,upd)
            critic={**critic,'net':optax.apply_updates(critic['net'],upd)}
    return key,critic,opt_v_state,critic_loss


def policy_update_impl(
    key, rollout_key, state0, xi0, xi_n, frame0, prev0, obs0, goal0,
    policy, critic, wm, opt_pi_state, opt_v_state, lr,
    *, mdp, motion, opt_pi, opt_v, horizon, pi_batch, gamma, lam,
    no_critic, critic_epochs, critic_minibatches, goal_frames,
):
    """Pure imagined rollout, GAE, and policy/critic optimizer update."""
    policy,opt_pi_state,imagined_loss,rollout=_policy_actor_stage(
        rollout_key,state0,xi0,xi_n,frame0,prev0,obs0,goal0,
        policy,critic,wm,opt_pi_state,lr,
        mdp=mdp,motion=motion,opt_pi=opt_pi,horizon=horizon,pi_batch=pi_batch,
        gamma=gamma,lam=lam,no_critic=no_critic,goal_frames=goal_frames,
    )
    if no_critic:
        return key,policy,critic,opt_pi_state,opt_v_state,imagined_loss,jnp.asarray(0.,jnp.float32)
    critic,x,y=_critic_data_stage(
        rollout,state0,xi_n,critic,mdp=mdp,horizon=horizon,pi_batch=pi_batch,
        gamma=gamma,lam=lam,
    )
    key,critic,opt_v_state,critic_loss=_critic_update_stage(
        key,critic,opt_v_state,x,y,lr,opt_v=opt_v,
        critic_epochs=critic_epochs,critic_minibatches=critic_minibatches,
    )
    return key,policy,critic,opt_pi_state,opt_v_state,imagined_loss,critic_loss


def run(args):
    root=Path(__file__).resolve().parents[1]
    xml=Path(args.xml or root/'data/menagerie_unitree_g1/scene_mjx.xml')
    motion_dir=Path(args.motion_dir or root/'data/origin_lafan1_g1')
    kin=JAXG1Kinematics(str(xml))
    motion=JAXMotion.from_csv_dir(motion_dir,kin) if list(motion_dir.glob('*.csv')) else JAXMotion(motion_dir,kin)
    sim=JAXMJXG1(xml,args.num_envs,args.backend); mdp=JAXMDP(kin,motion,jnp.asarray(default_joint_pos_list(),jnp.float32))
    replay=JAXReplay(args.buffer_steps,args.num_envs); key=jax.random.key(args.seed)
    wm=init_world_model(jax.random.fold_in(key,1),hidden=(args.hidden,)*args.wm_layers)
    policy=init_policy(jax.random.fold_in(key,2),99,73*args.goal_frames,args.hist,args.z_dim)
    critic=init_critic(jax.random.fold_in(key,3),184+73*args.goal_frames)
    opt=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(1.0)); opt_state=opt.init(wm['net'])
    opt_pi=optax.chain(optax.clip_by_global_norm(1.0),optax.clip(.1),optax.adam(1.0,b1=.9,b2=.95)); pi_params={'enc':policy['enc'],'dec':policy['dec']}; opt_pi_state=opt_pi.init(pi_params)
    opt_v=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(1.0)); opt_v_state=opt_v.init(critic['net'])
    env=JAXTrackingEnv(sim,motion,kin,mdp,gamma=args.gamma,seed=args.seed)
    state=state_from_data(sim.data); frame,xi,prev,ep_step=env.state.frame,env.state.xi,env.state.a_prev,env.state.ep_step
    obs_hist=jnp.zeros((args.num_envs,args.hist,99),jnp.float32)
    last_world_loss=None; last_policy_loss=None; last_critic_loss=None
    error_floor=jnp.asarray([3e-3]*3+[5e-3]*3+[2e-2]*3+[5e-2]*3+[3e-3]*29+[5e-2]*29,jnp.float32)
    compiled_wm_update=jax.jit(lambda params,opt_state,batch,lr: compiled_world_update(params,opt_state,batch,lr,opt,error_floor,args.wm_window))
    compiled_sample_starts=jax.jit(lambda data,ptr,size,key: sample_starts(data,ptr,size,key,args.pi_batch,args.hist,args.horizon))
    def wm_scan(carry, data, ptr, size, lr):
        params,opt_state,ks=carry; ks,kb=jax.random.split(ks)
        batch=sample_windows(data,ptr,size,kb,args.wm_batch,args.wm_window)
        params,opt_state,loss=compiled_world_update(params,opt_state,batch,lr,opt,error_floor,args.wm_window)
        return (params,opt_state,ks),loss
    def wm_scan_runner(params,opt_state,ks,data,ptr,size,lr):
        body=lambda carry,_: wm_scan(carry,data,ptr,size,lr)
        return jax.lax.scan(body,(params,opt_state,ks),None,length=args.wm_batches)
    compiled_wm_scan=jax.jit(wm_scan_runner)
    def prepare_policy_inputs(rollout_key,start,hist):
        state0=unflatten_state(start['s']); xi0=start['xi']; xi_n=normalize_xi(xi0)
        frame0=start['f']; prev0=start['a_prev']
        obs0=mdp.add_noise(
            mdp.obs(unflatten_state(hist['s']),hist['a_prev'],hist['xi'],hist['f']),
            jax.random.fold_in(rollout_key,1),
        )
        goal0=motion.goal(frame0,args.goal_frames)
        return state0,xi0,xi_n,frame0,prev0,obs0,goal0

    def policy_update_runner(key,rollout_key,start,hist,policy,critic,wm,opt_pi_state,opt_v_state,lr):
        inputs=prepare_policy_inputs(rollout_key,start,hist)
        return policy_update_impl(
            key,rollout_key,*inputs,policy,critic,wm,opt_pi_state,opt_v_state,lr,
            mdp=mdp,motion=motion,opt_pi=opt_pi,opt_v=opt_v,horizon=args.horizon,pi_batch=args.pi_batch,
            gamma=args.gamma,lam=args.lam,no_critic=args.no_critic,critic_epochs=args.critic_epochs,
            critic_minibatches=args.critic_minibatches,goal_frames=args.goal_frames,
        )
    compiled_policy_update=jax.jit(policy_update_runner)

    if args.profile_policy_stages:
        if args.no_critic:
            raise ValueError('--profile-policy-stages requires the critic')

        def actor_stage_runner(rollout_key,state0,xi0,xi_n,frame0,prev0,obs0,goal0,policy,critic,wm,opt_pi_state,lr):
            return _policy_actor_stage(
                rollout_key,state0,xi0,xi_n,frame0,prev0,obs0,goal0,policy,critic,wm,opt_pi_state,lr,
                mdp=mdp,motion=motion,opt_pi=opt_pi,horizon=args.horizon,pi_batch=args.pi_batch,
                gamma=args.gamma,lam=args.lam,no_critic=False,goal_frames=args.goal_frames,
            )

        def critic_data_runner(rollout,state0,xi_n,critic):
            return _critic_data_stage(
                rollout,state0,xi_n,critic,mdp=mdp,horizon=args.horizon,pi_batch=args.pi_batch,
                gamma=args.gamma,lam=args.lam,
            )

        def critic_update_runner(key,critic,opt_v_state,x,y,lr):
            return _critic_update_stage(
                key,critic,opt_v_state,x,y,lr,opt_v=opt_v,
                critic_epochs=args.critic_epochs,critic_minibatches=args.critic_minibatches,
            )

        compiled_actor_stage=jax.jit(actor_stage_runner)
        compiled_critic_data_stage=jax.jit(critic_data_runner)
        compiled_critic_update_stage=jax.jit(critic_update_runner)

        def time_profile_stage(name,fn,*inputs):
            t0=time.perf_counter(); output=fn(*inputs); _sync_tree(output)
            elapsed=time.perf_counter()-t0
            print(f'profile_stage_{name}_it_{it+1}_ms={1000*elapsed:.2f}'+_memory_stats_text(args.backend),flush=True)
            return output,elapsed

    def make_collect_scan(length):
        def body(carry, _):
            train_key, env_state, env_key, model_batch, data, phys, history, policy, wm = carry
            train_key,ka,ko=jax.random.split(train_key,3)
            goal=motion.goal(env_state.frame,args.goal_frames)
            obs=mdp.add_noise(mdp.obs(phys,env_state.a_prev,env_state.xi,env_state.frame),ko)
            policy={**policy,'obs_norm':update_norm(policy['obs_norm'],obs),'goal_norm':update_norm(policy['goal_norm'],goal)}
            fresh=env_state.ep_step==0
            history=jnp.where(fresh[:,None,None],obs[:,None,:],jnp.concatenate((history[:,1:],obs[:,None,:]),1))
            action=jnp.clip(policy_apply(policy,history,goal)+args.action_noise*jax.random.normal(ka,(args.num_envs,29)),-20,20)
            (env_state,env_key,model_batch,data,phys),tr=env.step_pure(action,(env_state,env_key,model_batch,data,phys))
            old_state=unflatten_state(tr['s']); next_state=unflatten_state(tr['s1'])
            wm={**wm,'in_norm':update_norm(wm['in_norm'],jnp.concatenate((invariant_features(old_state),action,normalize_xi(tr['xi'])),-1)),'out_norm':update_norm(wm['out_norm'],wm_target(old_state,next_state))}
            return (train_key,env_state,env_key,model_batch,data,phys,history,policy,wm),tr
        return jax.jit(lambda carry: jax.lax.scan(body,carry,None,length=length))

    burst_lengths={1,*[n for n in (math.floor(1/args.utd),math.ceil(1/args.utd),args.warmup_steps) if n>0]}
    collect_scans={n:make_collect_scan(n) for n in burst_lengths}

    def collect_batch(n):
        nonlocal key,state,frame,xi,prev,ep_step,obs_hist,policy,wm
        carry=(key,env.state,env.key,env.sim._model_batch,env.sim.data,env.phys_state,obs_hist,policy,wm)
        carry,batch=collect_scans[n](carry)
        key,env_state,env.key,env.sim._model_batch,env.sim._data,env.phys_state,obs_hist,policy,wm=carry
        env.state=env_state
        state=env.phys_state
        frame,xi,prev,ep_step=env.state.frame,env.state.xi,env.state.a_prev,env.state.ep_step
        return batch

    def collect_one():
        return jax.tree_util.tree_map(lambda x:x[0],collect_batch(1))

    def world_update():
        nonlocal key,wm,opt_state,last_world_loss
        (wm,opt_state,key),batch_losses=compiled_wm_scan(wm,opt_state,key,replay.data,jnp.asarray(replay.ptr,jnp.int32),jnp.asarray(replay.size,jnp.int32),jnp.asarray(lr_now,jnp.float32))
        last_world_loss=batch_losses[-1]

    def policy_update():
        nonlocal key,policy,critic,opt_pi_state,opt_v_state,last_policy_loss,last_critic_loss
        key,ks=jax.random.split(key); start,hist=compiled_sample_starts(replay.data,jnp.asarray(replay.ptr,jnp.int32),jnp.asarray(replay.size,jnp.int32),ks)
        lr=jnp.asarray(lr_now,jnp.float32)
        if args.profile_policy_stages:
            t0=time.perf_counter(); inputs=prepare_policy_inputs(ks,start,hist); _sync_tree(inputs)
            prep_s=time.perf_counter()-t0
            print(f'profile_stage_input_prep_it_{it+1}_ms={1000*prep_s:.2f}'+_memory_stats_text(args.backend),flush=True)
            state0,xi0,xi_n,frame0,prev0,obs0,goal0=inputs
            (policy,opt_pi_state,imagined_loss,rollout),actor_s=time_profile_stage(
                'actor_rollout_grad',compiled_actor_stage,ks,*inputs,policy,critic,wm,opt_pi_state,lr,
            )
            (critic,x,y),features_s=time_profile_stage(
                'critic_features_gae',compiled_critic_data_stage,rollout,state0,xi_n,critic,
            )
            (key,critic,opt_v_state,critic_loss),critic_s=time_profile_stage(
                'critic_updates',compiled_critic_update_stage,key,critic,opt_v_state,x,y,lr,
            )
            print(f'profile_stage_total_it_{it+1}_ms={1000*(prep_s+actor_s+features_s+critic_s):.2f}',flush=True)
        else:
            key,policy,critic,opt_pi_state,opt_v_state,imagined_loss,critic_loss=compiled_policy_update(
                key,ks,start,hist,policy,critic,wm,opt_pi_state,opt_v_state,lr,
            )
        last_policy_loss=imagined_loss
        last_critic_loss=critic_loss
        return imagined_loss,critic_loss

    # Keep profile warmup on the same one-step collection path as profile iterations.
    if args.profile:
        replay.add_many([collect_one() for _ in range(args.warmup_steps)])
    elif args.warmup_steps:
        replay.add_batch(collect_batch(args.warmup_steps))
    step_acc=0.0; imagined_loss=critic_loss=None
    profile_totals={'collect_s':0.0,'world_s':0.0,'policy_critic_s':0.0,'iteration_s':0.0}
    profile_count=0
    first_policy_profile_s=None; steady_policy_profile_s=0.0; steady_policy_profile_count=0
    if args.memory_log_every>0 and args.backend=='mps':
        print('phase learning start'+_memory_stats_text(args.backend),flush=True)
    def save_checkpoint(path):
        from .jax_checkpoint import save
        save(path,{'wm':wm,'policy':policy,'critic':critic,'opt_state':opt_state,'opt_pi_state':opt_pi_state,'opt_v_state':opt_v_state,'env_state':env.state,'result':{'iterations':it+1,'sim_steps':(args.warmup_steps+int((it+1)/args.utd))*args.num_envs,'replay':replay.size,'last_world_loss':float(last_world_loss) if last_world_loss is not None else None,'imagined_policy_loss':float(imagined_loss) if imagined_loss is not None else None,'critic_loss':float(critic_loss) if critic_loss is not None else None,'backend':jax.default_backend(),'torch_free':True},'config':vars(args)})
    for it in range(args.steps):
        iter_t=time.perf_counter()
        lr_now=args.lr*(.03+.97*.5*(1+jnp.cos(jnp.pi*min(it,args.lr_decay_iters)/args.lr_decay_iters))) if it<args.lr_decay_iters else args.lr*.03
        step_acc+=1/args.utd; n=int(step_acc); step_acc-=n
        phase_t=time.perf_counter()
        if args.profile:
            replay.add_many([collect_one() for _ in range(n)])
        elif n:
            replay.add_batch(collect_batch(n))
        if args.profile: _sync_tree((state,frame,xi,prev,ep_step,obs_hist,wm,replay.data))
        collect_s=time.perf_counter()-phase_t
        world_s=policy_s=0.0
        if args.wm_batches>0 and replay.size>=args.wm_window:
            phase_t=time.perf_counter(); world_update()
            if args.profile: _sync_tree((wm,opt_state,key,last_world_loss))
            world_s=time.perf_counter()-phase_t
            if args.profile:
                print(f'profile_ms_world_update_it_{it+1}='+str(round(1000*world_s,2)),flush=True)
        if it>=args.wm_only_iters and replay.size>=args.horizon:
            phase_t=time.perf_counter(); imagined_loss,critic_loss=policy_update()
            if args.profile: _sync_tree((policy,critic,opt_pi_state,opt_v_state,key,imagined_loss,critic_loss))
            policy_s=time.perf_counter()-phase_t
            if args.profile:
                print(f'profile_ms_policy_update_it_{it+1}='+str(round(1000*policy_s,2)),flush=True)
                if first_policy_profile_s is None:
                    first_policy_profile_s=policy_s
                else:
                    steady_policy_profile_s+=policy_s
                    steady_policy_profile_count+=1
        iteration_s=time.perf_counter()-iter_t
        if args.memory_log_every>0 and (it+1)%args.memory_log_every==0:
            print(f'phase learning it {it+1}'+_memory_stats_text(args.backend),flush=True)
        if args.profile and it>=args.profile_warmup_iters:
            profile_totals['collect_s']+=collect_s; profile_totals['world_s']+=world_s; profile_totals['policy_critic_s']+=policy_s; profile_totals['iteration_s']+=iteration_s; profile_count+=1
        if it+1==args.steps or (args.log_every>0 and (it+1)%args.log_every==0):
            print(f"it {it+1} | replay {replay.size}/{args.buffer_steps} | sim_steps {(args.warmup_steps+int((it+1)/args.utd))*args.num_envs} | wm {float(last_world_loss) if last_world_loss is not None else float('nan'):.4f} | pi {float(imagined_loss) if imagined_loss is not None else float('nan'):.4f} | v {float(critic_loss) if critic_loss is not None else float('nan'):.4f}",flush=True)
        if args.checkpoint and args.checkpoint_every>0 and (it+1)%args.checkpoint_every==0:
            save_checkpoint(Path(args.checkpoint)/f"iter_{it+1:06d}")
    if args.profile and profile_count:
        print('profile_ms_per_iteration='+str({k:round(1000*v/profile_count,2) for k,v in profile_totals.items()}),flush=True)
        if first_policy_profile_s is not None:
            print('profile_ms_first_policy_update='+str(round(1000*first_policy_profile_s,2)),flush=True)
        if steady_policy_profile_count:
            print('profile_ms_steady_policy_update='+str(round(1000*steady_policy_profile_s/steady_policy_profile_count,2)),flush=True)
    result={'iterations':args.steps,'sim_steps':(args.warmup_steps+int(args.steps/args.utd))*args.num_envs,'replay':replay.size,'last_world_loss':float(last_world_loss) if last_world_loss is not None else None,'imagined_policy_loss':float(imagined_loss) if imagined_loss is not None else None,'critic_loss':float(critic_loss) if critic_loss is not None else None,'backend':jax.default_backend(),'torch_free':True}
    if args.checkpoint:
        from .jax_checkpoint import save
        save(args.checkpoint,{'wm':wm,'policy':policy,'critic':critic,'opt_state':opt_state,'opt_pi_state':opt_pi_state,'opt_v_state':opt_v_state,'env_state':env.state,'result':result,'config':vars(args)})
    print(result)
    return last_world_loss


def main():
    p=argparse.ArgumentParser(); p.add_argument('--backend',default='mps',choices=('mps','cpu')); p.add_argument('--steps',type=int,default=8); p.add_argument('--iterations',type=int,default=None); p.add_argument('--utd',type=float,default=.4); p.add_argument('--warmup-steps',type=int,default=64); p.add_argument('--wm-only-iters',type=int,default=500); p.add_argument('--wm-batches',type=int,default=10); p.add_argument('--num-envs',type=int,default=8); p.add_argument('--buffer-steps',type=int,default=32); p.add_argument('--wm-window',type=int,default=8); p.add_argument('--wm-batch',type=int,default=4); p.add_argument('--hidden',type=int,default=64); p.add_argument('--wm-layers',type=int,default=2); p.add_argument('--hist',type=int,default=10); p.add_argument('--goal-frames',type=int,default=10); p.add_argument('--horizon',type=int,default=32); p.add_argument('--pi-batch',type=int,default=4); p.add_argument('--z-dim',type=int,default=32); p.add_argument('--gamma',type=float,default=.99); p.add_argument('--lam',type=float,default=.95); p.add_argument('--action-noise',type=float,default=.1); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--lr-decay-iters',type=int,default=250000); p.add_argument('--critic-epochs',type=int,default=5); p.add_argument('--critic-minibatches',type=int,default=4); p.add_argument('--no-critic',action='store_true'); p.add_argument('--log-every',type=int,default=50); p.add_argument('--memory-log-every',type=int,default=0); p.add_argument('--profile',action='store_true'); p.add_argument('--profile-policy-stages',action='store_true'); p.add_argument('--profile-warmup-iters',type=int,default=1); p.add_argument('--checkpoint'); p.add_argument('--checkpoint-every',type=int,default=0); p.add_argument('--seed',type=int,default=0); p.add_argument('--motion-dir'); p.add_argument('--xml'); args=p.parse_args(); args.steps=args.iterations if args.iterations is not None else args.steps; run(args)

if __name__=='__main__': main()
