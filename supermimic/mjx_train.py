"""Apple Silicon MJX implementation of the original SuperMimic training loop."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .buffer import ReplayBuffer
from .kinematics import G1Kinematics
from .mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, STATE_DIM, XI_DIM, flatten_state, normalize_xi, unflatten_state
from .motion import MotionLib
from .mjx_backend import MJXG1Simulator
from .mjx_env import MJXTrackingEnv
from .networks import Critic, Policy, RunningNorm, WorldModel
from .robot import default_joint_pos_list


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")
    return torch.device(name)


def _set_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _evaluate(env: MJXTrackingEnv, motion: MotionLib, mdp: MDP, pi: Policy, horizon: int) -> dict:
    clips = len(motion.names)
    if env.N != clips:
        raise ValueError("evaluation environment count must equal the number of motion clips")
    ids = torch.arange(clips, device=env.device)
    env.done_eval.zero_(); env.failed_eval.zero_()
    env.reset(ids, motion.clip_start.clone())
    returns = torch.zeros(clips, device=env.device)
    hist = None
    while not bool(env.done_eval.all()):
        state = env.state()
        obs = mdp.obs(state, env.a_prev, env.xi, env.f)
        hist = obs[:, None].expand(-1, horizon, -1).clone() if hist is None else torch.cat((hist[:, 1:], obs[:, None]), 1)
        action = pi(hist, motion.goal(env.f, horizon))
        alive = ~env.done_eval
        out = env.step(action)
        returns += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, motion.clip_end[motion.clip_id[env.f]] - 1)
    norm = returns / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    return {"norm_return": float(norm.mean()), "norm_return_framew": float(returns.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)),
            "n_failed": int(env.failed_eval.sum()), "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
            "per_clip": {name: round(float(norm[i]), 4) for i, name in enumerate(motion.names)}}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = _device(args.device)
    root = Path(__file__).resolve().parents[1]
    xml_path = Path(args.xml or root / "data/g1_mjcf/scene_flat.xml").resolve()
    motion_dir = Path(args.motion_dir or root / "data/g1_motions").resolve()
    kin = G1Kinematics(str(device), xml_path=str(xml_path))
    motion = MotionLib.from_npz_dir(str(motion_dir), kin, str(device))
    default_q = torch.tensor(default_joint_pos_list(), device=device)
    mdp = MDP(kin, motion, default_q)
    sim = MJXG1Simulator(xml_path, args.num_envs, backend=args.backend)
    env = MJXTrackingEnv(sim, motion, mdp, gamma=args.gamma, reset_noise=True)
    goal_dim = 73 * args.goal_frames
    wm = WorldModel(hidden=(args.hidden,) * args.wm_layers).to(device)
    pi = Policy(OBS_DIM, goal_dim, hist=args.hist, z_dim=args.z_dim).to(device)
    critic = Critic(INV_DIM + CRITIC_EXTRA + goal_dim).to(device)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_steps, args.num_envs, str(device))
    obs_hist = torch.zeros(args.num_envs, args.hist, OBS_DIM, device=device)
    critic_norm = RunningNorm(INV_DIM + CRITIC_EXTRA + goal_dim).to(device)
    error_floor = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2),
                             torch.full((3,), 5e-2), torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(device)
    start_iteration, sim_steps, losses = 0, 0, []
    out_dir = Path(args.log_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        contract = {"state_dim": STATE_DIM, "xi_dim": XI_DIM, "obs_dim": OBS_DIM, "action_dim": 29}
        if ck.get("contract") != contract:
            raise ValueError("checkpoint dimensions do not match the SuperMimic contract")
        for module, key in ((wm,"wm"),(pi,"pi"),(critic,"critic"),(critic_norm,"critic_norm")):
            module.load_state_dict(ck[key])
        for optimizer, key in ((opt_wm,"opt_wm"),(opt_pi,"opt_pi"),(opt_v,"opt_v")):
            optimizer.load_state_dict(ck[key])
        replay.load_state_dict(ck["replay"])
        start_iteration, sim_steps = int(ck["iteration"]), int(ck["sim_steps"])

    @torch.no_grad()
    def collect(steps: int) -> None:
        nonlocal obs_hist, sim_steps
        for _ in range(steps):
            state = env.state(); frame = env.f.clone(); xi = env.xi.clone(); previous = env.a_prev.clone()
            obs = mdp.add_noise(mdp.obs(state, previous, xi, frame))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], obs[:, None].expand_as(obs_hist), torch.cat((obs_hist[:, 1:], obs[:, None]), 1))
            goal = motion.goal(frame, args.goal_frames)
            pi.obs_norm.update(obs); pi.goal_norm.update(goal)
            action = (pi(obs_hist, goal) + args.action_noise * torch.randn_like(previous)).clamp(-20, 20)
            transition = env.step(action)
            transition.update({"clip_end": motion.clip_end[motion.clip_id[frame]], "ep_step": transition["ep_step"].to(torch.int32)})
            replay.add(transition)
            s0, s1 = unflatten_state(transition["s"]), unflatten_state(transition["s1"])
            wm.in_norm.update(wm.inputs(s0, action, normalize_xi(xi)))
            wm.out_norm.update(wm.targets(s0, s1))
            sim_steps += args.num_envs
            done = transition["done"]
            if bool(done.any()):
                obs_hist[done] = 0
                env.a_prev[done] = 0

    def wm_update() -> torch.Tensor:
        batch = replay.sample_windows(args.wm_batch, args.wm_window)
        state = unflatten_state(batch["s"][:, 0]); errors = []
        for t in range(args.wm_window):
            state = wm.step(state, batch["a"][:, t], normalize_xi(batch["xi"][:, t]))
            errors.append(wm.state_error(state, unflatten_state(batch["s1"][:, t])))
        error = torch.stack(errors, 1)
        loss = ((error / torch.maximum(wm.out_norm.std(), error_floor)).square().mean(-1) * batch["mask"]).sum() / args.wm_batch
        opt_wm.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt_wm.step()
        return loss.detach() / args.wm_window

    def policy_update() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = replay.sample_starts(args.pi_batch, args.hist, args.horizon)
        state = unflatten_state(start["s"]); state0 = state
        xi, xi_n, frame, previous = start["xi"], normalize_xi(start["xi"]), start["f"], start["a_prev"]
        obs_history = mdp.add_noise(mdp.obs(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]))
        goal0 = motion.goal(frame, args.goal_frames); goal = goal0
        _set_grad(wm, False); _set_grad(critic, False)
        states, actions, frames, goals, rewards, dones = [], [], [], [], [], []
        for _ in range(args.horizon):
            action = pi(obs_history, goal)
            state = wm.step(state, action, xi_n); frame = frame + 1
            goal = motion.goal(frame, args.goal_frames)
            obs = mdp.add_noise(mdp.obs(state, action, xi, frame))
            obs_history = torch.cat((obs_history[:, 1:], obs[:, None]), 1)
            reward, done = mdp.reward(state, frame, action, previous), mdp.soft_termination(state, frame)
            states.append(state); actions.append(action); frames.append(frame); goals.append(goal); rewards.append(reward); dones.append(done)
            previous = action
        alive = torch.cumprod(1-torch.stack(dones), 0)
        continuation = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discount = args.gamma ** torch.arange(args.horizon, device=device).float()[:, None]
        returns = (discount * continuation * torch.stack(rewards)).sum(0)
        returns += args.gamma**args.horizon * alive[-1] * critic(torch.cat((mdp.critic_features(state, frame, xi_n), goal), -1))
        pi_loss = -returns.mean(); opt_pi.zero_grad(set_to_none=True); pi_loss.backward()
        grad = nn.utils.clip_grad_norm_(pi.parameters(), 1.0); nn.utils.clip_grad_value_(pi.parameters(), .1)
        if torch.isfinite(grad): opt_pi.step()
        _set_grad(wm, True); _set_grad(critic, True)
        with torch.no_grad():
            states_all = tuple(torch.cat((state0[i][None], torch.stack([s[i] for s in states]),), 0).flatten(0,1) for i in range(6))
            frames_all = torch.cat((start["f"][None], torch.stack(frames)), 0).flatten()
            goals_all = torch.cat((goal0[None], torch.stack(goals)), 0).flatten(0,1)
            xi_all = xi_n[None].expand(args.horizon+1,-1,-1).flatten(0,1)
            features = torch.cat((mdp.critic_features(states_all, frames_all, xi_all), goals_all), -1).view(args.horizon+1,args.pi_batch,-1)
        critic_norm.update(features)
        with torch.no_grad():
            values = critic(features).squeeze(-1); gae = torch.zeros_like(values[0]); targets=[]
            for t in range(args.horizon-1,-1,-1):
                delta = rewards[t].detach() + args.gamma*(1-dones[t].detach())*values[t+1] - values[t]
                gae = delta + args.gamma*args.lam*(1-dones[t].detach())*gae
                targets.append(gae+values[t])
            targets = torch.stack(targets[::-1]).reshape(-1)
        critic_inputs = features[:-1].reshape(-1,features.shape[-1]).detach(); n=critic_inputs.shape[0]
        for _ in range(args.critic_epochs):
            for indices in torch.tensor_split(torch.randperm(n,device=device), min(args.critic_minibatches,n)):
                v_loss = nn.functional.mse_loss(critic(critic_inputs[indices]).squeeze(-1),targets[indices])
                opt_v.zero_grad(set_to_none=True); v_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),1.0); opt_v.step()
        return pi_loss.detach(), v_loss.detach(), returns.detach().mean()

    collect(args.warmup_steps)
    step_acc = 0.0
    for iteration in range(start_iteration, start_iteration + args.iterations):
        lr = args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(iteration,args.lr_decay_iters) / args.lr_decay_iters)))
        for optimizer in (opt_wm,opt_pi,opt_v):
            for group in optimizer.param_groups: group["lr"] = lr
        step_acc += 1 / args.utd; nsteps = int(step_acc); step_acc -= nsteps
        t0=time.perf_counter(); collect(nsteps); collect_s=time.perf_counter()-t0
        if replay.size < max(args.wm_window+1,args.wm_batch,args.pi_batch):
            continue
        t1=time.perf_counter()
        for _ in range(args.wm_batches): loss=wm_update()
        if iteration >= args.wm_only_iters:
            pi_loss,v_loss,imag_return=policy_update()
        else:
            pi_loss=v_loss=imag_return=torch.zeros((),device=device)
        if device.type == "mps": torch.mps.synchronize()
        update_s=time.perf_counter()-t1; losses.append(float(loss.cpu()))
        if (iteration+1) % args.log_every == 0 or iteration == start_iteration:
            print(f"iter {iteration+1} | replay {replay.size} | wm {losses[-1]:.5f} | pi {float(pi_loss):.5f} | critic {float(v_loss):.5f} | imag {float(imag_return):.5f} | collect {collect_s:.3f}s update {update_s:.3f}s | {device}/{args.backend}",flush=True)
        if (iteration+1) % args.ckpt_every == 0 or iteration+1 == start_iteration+args.iterations:
            ck={"wm":wm.state_dict(),"pi":pi.state_dict(),"critic":critic.state_dict(),"critic_norm":critic_norm.state_dict(),
                "opt_wm":opt_wm.state_dict(),"opt_pi":opt_pi.state_dict(),"opt_v":opt_v.state_dict(),"iteration":iteration+1,
                "sim_steps":sim_steps,"args":vars(args),"replay":replay.state_dict(),
                "contract":{"state_dim":STATE_DIM,"xi_dim":XI_DIM,"obs_dim":OBS_DIM,"action_dim":29}}
            torch.save(ck,out_dir/"last.pt")
    if args.eval_after:
        eval_env = MJXTrackingEnv(MJXG1Simulator(xml_path,len(motion.names),args.backend),motion,mdp,gamma=args.gamma,eval_mode=True,reset_noise=False)
        pi.eval(); result=_evaluate(eval_env,motion,mdp,pi,args.goal_frames)
        (out_dir/"eval.json").write_text(json.dumps(result,indent=2)+"\n")
        print("evaluation",json.dumps({k:v for k,v in result.items() if k!="per_clip"}),flush=True)
    return {"iterations":args.iterations,"sim_steps":sim_steps,"last_world_loss":losses[-1] if losses else None,"device":str(device),"mjx_backend":args.backend}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=("mps","cpu"),default="mps")
    parser.add_argument("--device",choices=("auto","mps","cpu"),default="mps")
    parser.add_argument("--num-envs",type=int,default=256); parser.add_argument("--iterations",type=int,default=1000)
    parser.add_argument("--buffer-steps",type=int,default=1000); parser.add_argument("--utd",type=float,default=.4)
    parser.add_argument("--warmup-steps",type=int,default=64); parser.add_argument("--wm-window",type=int,default=8)
    parser.add_argument("--wm-batch",type=int,default=2048); parser.add_argument("--wm-batches",type=int,default=10)
    parser.add_argument("--pi-batch",type=int,default=4096); parser.add_argument("--goal-frames",type=int,default=10)
    parser.add_argument("--hist",type=int,default=10); parser.add_argument("--horizon",type=int,default=32)
    parser.add_argument("--hidden",type=int,default=2048); parser.add_argument("--wm-layers",type=int,default=3)
    parser.add_argument("--z-dim",type=int,default=32); parser.add_argument("--lr",type=float,default=2e-4)
    parser.add_argument("--lr-decay-iters",type=int,default=250000); parser.add_argument("--gamma",type=float,default=.99)
    parser.add_argument("--lam",type=float,default=.95); parser.add_argument("--action-noise",type=float,default=.1)
    parser.add_argument("--critic-epochs",type=int,default=5); parser.add_argument("--critic-minibatches",type=int,default=4)
    parser.add_argument("--wm-only-iters",type=int,default=500); parser.add_argument("--log-every",type=int,default=10)
    parser.add_argument("--ckpt-every",type=int,default=2000); parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--motion-dir",default=None)
    parser.add_argument("--xml",default=None); parser.add_argument("--log-dir",default="logs/mjx")
    parser.add_argument("--resume",default=None); parser.add_argument("--eval-after",action="store_true")
    args=parser.parse_args()
    print(json.dumps(train(args),indent=2))


if __name__ == "__main__":
    main()
