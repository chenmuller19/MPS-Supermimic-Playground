"""SuperMimic tracking environment backed by batched MJX G1 dynamics."""

from __future__ import annotations

import numpy as np
import jax
import torch

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import MotionLib
from .mjx_backend import MJXG1Simulator
from .rotations import mat_to_quat, so3_exp
from .robot import G1_JOINT_NAMES, action_scale


class MJXTrackingEnv:
    """MJX implementation of the original ``TrackingSim`` transition contract."""

    def __init__(
        self,
        sim: MJXG1Simulator,
        motion: MotionLib,
        mdp: MDP,
        gamma: float = 0.99,
        eval_mode: bool = False,
        max_episode_steps: int = 500,
        reset_noise: bool = True,
    ) -> None:
        self.sim, self.motion, self.mdp = sim, motion, mdp
        self.device = mdp.default_q.device
        self.N = sim.num_envs
        self.gamma = gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = mdp.default_q
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=self.device)
        self.xi = torch.zeros(self.N, 33, device=self.device)
        self.xi[:, 0] = 1.0
        self.f = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(self.N, 29, device=self.device)
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros(self.N, device=self.device)
        self.ep_ret = torch.zeros(self.N, device=self.device)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=self.device)
        self.bin_w_avg = torch.full((motion.num_bins,), 1 / motion.num_bins, device=self.device)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self.reset(torch.arange(self.N, device=self.device))

    def state(self):
        return self._state

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.clip_id[start]])
        frames = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return frames, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None) -> None:
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        if frames is None:
            frames, bins = self.sample_frames(count)
        else:
            frames = frames.to(self.device)
            bins = self.motion.frame_bin[frames]
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        xi = torch.zeros(count, 33, device=self.device)
        xi[:, 0] = 1.0
        if self.reset_noise:
            p += torch.stack((torch.empty(count, device=self.device).uniform_(-.05,.05),
                              torch.empty(count, device=self.device).uniform_(-.05,.05),
                              torch.empty(count, device=self.device).uniform_(-.01,.01)), -1)
            R = so3_exp(torch.stack((torch.empty(count, device=self.device).uniform_(-.1,.1),
                                     torch.empty(count, device=self.device).uniform_(-.1,.1),
                                     torch.empty(count, device=self.device).uniform_(-.2,.2)), -1)) @ R
            v += torch.stack((torch.empty(count, device=self.device).uniform_(-.5,.5),
                              torch.empty(count, device=self.device).uniform_(-.5,.5),
                              torch.empty(count, device=self.device).uniform_(-.2,.2)), -1)
            w += torch.stack((torch.empty(count, device=self.device).uniform_(-.52,.52),
                              torch.empty(count, device=self.device).uniform_(-.52,.52),
                              torch.empty(count, device=self.device).uniform_(-.78,.78)), -1)
            q += torch.empty_like(q).uniform_(-.1,.1)
            xi[:, 1:30] = torch.empty(count, 29, device=self.device).uniform_(-.01,.01)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        qpos_new = torch.from_numpy(np.array(jax.device_get(qpos), copy=True)).to(self.device)
        qvel_new = torch.from_numpy(np.array(jax.device_get(qvel), copy=True)).to(self.device)
        qpos_new[env_ids, :3] = p
        qpos_new[env_ids, 3:7] = quat
        qpos_new[env_ids, 7:36] = q
        qvel_new[env_ids, :3] = v
        qvel_new[env_ids, 3:6] = w
        qvel_new[env_ids, 6:35] = qd
        self._state = tuple(value.to(self.device) for value in self.sim.set_state(qpos_new, qvel_new).state)
        self.xi[env_ids] = xi
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0
        self.ep_ret[env_ids] = 0

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict:
        S0 = self.state()
        f0, xi, a_prev, ep_step = self.f.clone(), self.xi.clone(), self.a_prev.clone(), self.ep_step.clone()
        result = self.sim.step(action, joint_offset=xi[:, 1:30])
        S1 = tuple(value.to(self.device) for value in result.state)
        self._state = S1
        f1 = f0 + 1
        reward, terms = self.mdp.reward(S1, f1, action, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f0]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {"s": flatten_state(S0), "a": action.clone(), "a_prev": a_prev, "f": f0,
               "s1": flatten_state(S1), "r": reward, "term": term, "done": done,
               "ep_step": ep_step, "xi": xi, "terms": terms}
        self.ep_disc_ret += self.gamma**ep_step.float() * reward
        self.ep_ret += reward
        self.f, self.ep_step, self.a_prev = f1, ep_step + 1, action.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
        else:
            ids = done.nonzero(as_tuple=False).flatten()
            if ids.numel():
                T = self.ep_step[ids].float()
                boot = torch.where(term[ids], 0.0, self.gamma**T * (self.ep_ret[ids] / T) / (1-self.gamma))
                returns = self.ep_disc_ret[ids] + boot
                b = self.start_bin[ids]
                self.bin_G.index_put_((b,), .8*self.bin_G[b] + .2*returns)
                out["ep_len"] = T
                out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
                self.reset(ids)
        return out
