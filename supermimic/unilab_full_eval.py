"""Evaluate a full SuperMimic checkpoint through the UniLab G1 simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .kinematics import G1Kinematics
from .mdp import MDP, OBS_DIM, MAX_STEP_REWARD
from .networks import Policy
from . import networks as network_module
from .rotations import quat_to_mat
from .unilab_adapter import UniLabMotionAdapter
from .unilab_full_train import device_for, make_env


def _make_reference(adapter, device: torch.device):
    class Reference:
        root_p = adapter.ref_body_pos_w[:, adapter.root_body_index].to(device)
        root_R = quat_to_mat(adapter.ref_body_quat_w[:, adapter.root_body_index]).to(device)
        body_p = adapter.ref_body_pos_w.to(device)
        body_R = quat_to_mat(adapter.ref_body_quat_w).to(device)
        body_v = adapter.ref_body_lin_vel_w.to(device)
        body_w = adapter.ref_body_ang_vel_w.to(device)
        q = adapter.ref_joint_pos.to(device)
        qd = adapter.ref_joint_vel.to(device)
    return Reference()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unilab-root", default="../UniLab-main")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2000)
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck["args"]
    dev = device_for(args.device)
    if cfg.get("amp") == "bf16":
        network_module.AMP_DTYPE = torch.bfloat16
    train_args = SimpleNamespace(unilab_root=args.unilab_root, sim="mujoco", num_envs=args.num_envs, iterations=1)
    env = make_env(train_args, eval_mode=True)
    try:
        env.init_state(); adapter = UniLabMotionAdapter(env)
        motion = _make_reference(adapter, dev)
        kin = G1Kinematics(str(dev), xml_path=str(Path(args.unilab_root).resolve() / "src/unilab/assets/robots/g1/scene_flat.xml"))
        mdp = MDP(kin, motion, adapter.default_q[0].to(dev))
        policy = Policy(OBS_DIM, 73 * int(cfg["goal_frames"]), hist=int(cfg["hist"]), z_dim=int(cfg.get("z_dim", 32))).to(dev)
        policy.load_state_dict(ck["pi"]); policy.eval()
        a_prev = torch.zeros(args.num_envs, 29, device=dev)
        hist = torch.zeros(args.num_envs, int(cfg["hist"]), OBS_DIM, device=dev)
        ret = torch.zeros(args.num_envs, device=dev); sm_ret = torch.zeros_like(ret)
        lengths = torch.zeros(args.num_envs, device=dev); done_count=0
        failures = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
        finished = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
        initialized = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
        for _ in range(args.max_steps):
            if bool(finished.all()):
                break
            S = tuple(x.to(dev) for x in adapter.state()); f_cpu=adapter.frame; f=f_cpu.to(dev); ce=adapter.clip_end.to(dev); xi=adapter.xi().to(dev)
            o=mdp.obs(S,a_prev,xi,f); g=adapter.future_goal(f_cpu,int(cfg["goal_frames"]),adapter.clip_end).to(dev)
            hist=torch.where(initialized[:,None,None], torch.cat((hist[:,1:],o[:,None]),1), o[:,None].expand_as(hist))
            initialized[:] = True
            with torch.no_grad(): action=policy(hist,g).clamp(-20,20)
            out=env.step(adapter.to_unilab_action(action).to(env.device)); r=out.reward.to(dev); done=(out.terminated|out.truncated).to(dev)
            next_state = tuple(value.to(dev) for value in adapter.state())
            next_frame = torch.minimum(f + 1, ce)
            sm_reward = mdp.reward(next_state, next_frame, action, a_prev)
            active = ~finished
            ret += torch.where(active, r, torch.zeros_like(r))
            sm_ret += torch.where(active, sm_reward, torch.zeros_like(sm_reward))
            lengths += active.float(); a_prev=action
            clip_complete = f + 1 >= ce
            done |= clip_complete
            if bool(done.any()):
                newly_done = done & ~finished
                done_count += int(newly_done.sum()); finished |= newly_done
                failures |= newly_done & (out.terminated.to(dev) | (out.truncated.to(dev) & ~clip_complete))
                ids=done.nonzero().flatten().to(env.device)
                if ids.numel():
                    env.reset(ids); a_prev=a_prev.clone(); a_prev[done]=0; hist=hist.clone(); hist[done]=0; initialized[done]=False
        result={"device":str(dev),"max_steps":args.max_steps,"reward_contract":"unilab_task_reward_is_diagnostic_only","mean_unilab_return":float(ret.mean().cpu()),"mean_supermimic_mdp_return":float(sm_ret.mean().cpu()),"mean_length":float(lengths.mean().cpu()),"done_events":done_count,"failed_clips":int(failures.sum().cpu()),"completed_clips":int((finished & ~failures).sum().cpu()),"incomplete_clips":int((~finished).sum().cpu()),"per_env":[{"env":i,"unilab_return":float(ret[i].cpu()),"supermimic_mdp_return":float(sm_ret[i].cpu()),"length":float(lengths[i].cpu()),"supermimic_mean_reward_fraction":float((sm_ret[i] / (lengths[i].clamp_min(1) * MAX_STEP_REWARD)).cpu()),"failed":bool(failures[i]),"completed":bool(finished[i] and not failures[i]),"incomplete":bool(not finished[i])} for i in range(args.num_envs)],"state_dim":71,"xi_dim":33,"obs_dim":OBS_DIM}
        print(json.dumps(result,indent=2))
    finally:
        env.close()

if __name__ == "__main__":
    main()
