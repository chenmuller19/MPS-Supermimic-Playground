"""Pure-JAX equivalents of the SuperMimic MLPs and running normalizers."""

from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp

from .jax_mjx_backend import NUM_J

XI_DIM = 33
INV_DIM = 1 + 6 + 3 + 3 + 29 + 29
WM_OUT = 70


class NormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_norm(width: int) -> NormState:
    return NormState(jnp.zeros((width,), jnp.float32), jnp.ones((width,), jnp.float32), jnp.array(0.0))


def update_norm(state: NormState, values: jax.Array, max_count: float = 2e8) -> NormState:
    x = values.reshape((-1, values.shape[-1])).astype(jnp.float32)
    n = jnp.asarray(x.shape[0], jnp.float32)
    mean = jnp.mean(x, axis=0)
    var = jnp.var(x, axis=0)
    total = state.count + n
    delta = mean - state.mean
    new_mean = state.mean + delta * n / jnp.maximum(total, 1.0)
    new_var = (state.var * state.count + var * n + delta**2 * state.count * n / jnp.maximum(total, 1.0)) / jnp.maximum(total, 1.0)
    return jax.lax.cond(state.count >= max_count, lambda _: state, lambda _: NormState(new_mean, new_var, total), operand=None)


def normalize(state: NormState, x: jax.Array, eps: float = 1e-2) -> jax.Array:
    return (x - state.mean) / jnp.sqrt(state.var + eps**2)


def _init_mlp(key, dims, *, layer_norm=False, final_zero=False, small_final=False):
    keys = jax.random.split(key, len(dims)-1); layers=[]
    for i, (din,dout,k) in enumerate(zip(dims[:-1],dims[1:],keys)):
        bound=1.0/jnp.sqrt(din); kw,kb=jax.random.split(k)
        w=jax.random.uniform(kw,(din,dout),minval=-bound,maxval=bound,dtype=jnp.float32); b=jax.random.uniform(kb,(dout,),minval=-bound,maxval=bound,dtype=jnp.float32)
        if i==len(dims)-2 and final_zero: w=jnp.zeros_like(w); b=jnp.zeros_like(b)
        elif i==len(dims)-2 and small_final: w=w*.01; b=jnp.zeros_like(b)
        layer={'w':w,'b':b}
        if layer_norm and i<len(dims)-2: layer['scale']=jnp.ones((dout,),jnp.float32); layer['shift']=jnp.zeros((dout,),jnp.float32)
        layers.append(layer)
    return tuple(layers)


def _mlp(params,x,*,activation='elu',layer_norm=False):
    for i,layer in enumerate(params):
        x=x@layer['w']+layer['b']
        if i<len(params)-1:
            if layer_norm:
                mean=jnp.mean(x,axis=-1,keepdims=True); var=jnp.mean((x-mean)**2,axis=-1,keepdims=True); x=(x-mean)*jax.lax.rsqrt(var+1e-5)*layer['scale']+layer['shift']
            x=jax.nn.silu(x) if activation=='silu' else jax.nn.elu(x)
    return x


def init_world_model(key, hidden=(2048,2048,2048)):
    dims=(INV_DIM+NUM_J+XI_DIM,*hidden,WM_OUT)
    return {"net":_init_mlp(key,dims,layer_norm=True,final_zero=True),
            "in_norm":init_norm(dims[0]),"out_norm":init_norm(WM_OUT)}


def world_model_apply(params, features):
    x=normalize(params["in_norm"],features)
    y=_mlp(params["net"],x,activation="silu",layer_norm=True)
    std=jnp.sqrt(params["out_norm"].var+1e-8)
    return y*std+params["out_norm"].mean


def init_policy(key, obs_dim, goal_dim, hist=10, z_dim=32):
    k1,k2=jax.random.split(key)
    return {"enc":_init_mlp(k1,(goal_dim,2048,1024,512,256,z_dim)),
            "dec":_init_mlp(k2,(z_dim+obs_dim*hist,2048,1024,512,256,NUM_J),small_final=True),
            "obs_norm":init_norm(obs_dim),"goal_norm":init_norm(goal_dim)}


def policy_apply(params, obs_hist, goal):
    z=jnp.tanh(_mlp(params["enc"],normalize(params["goal_norm"],goal)))
    obs=normalize(params["obs_norm"],obs_hist).reshape((obs_hist.shape[0],-1))
    return _mlp(params["dec"],jnp.concatenate((z,obs),-1))


def init_critic(key, in_dim):
    return {"net":_init_mlp(key,(in_dim,256,128,1)),"norm":init_norm(in_dim)}


def critic_apply(params,x):
    return _mlp(params["net"],normalize(params["norm"],x)).squeeze(-1)
