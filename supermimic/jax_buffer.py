"""Fixed-shape JAX replay storage for the native learner."""

from __future__ import annotations

import jax
import jax.numpy as jnp

STATE_DIM=71; XI_DIM=33


def replay_add(data, ptr, size, transition):
    """Functional fixed-shape ring-buffer write for jax.jit/lax.scan."""
    data={key:value.at[ptr].set(transition[key]) for key,value in data.items()}
    return data, (ptr+1)%data['s'].shape[0], jnp.minimum(size+1,data['s'].shape[0])


def sample_windows(data, ptr, size, key, batch_size, window):
    k0,k1=jax.random.split(key); T=data['s'].shape[0]; N=data['s'].shape[1]
    max_t=jnp.maximum(size-window+1,1); t0=jax.random.randint(k0,(batch_size,),0,max_t); env=jax.random.randint(k1,(batch_size,),0,N)
    ages=t0[:,None]+jnp.arange(window); rows=(ptr-size+ages)%T; e=jnp.broadcast_to(env[:,None],rows.shape)
    out={name:data[name][rows,e] for name in ('s','s1','a','xi','done')}; d=out['done'].astype(jnp.float32)
    out['mask']=jnp.cumprod(jnp.concatenate((jnp.ones((batch_size,1)),1-d[:,:-1]),-1),-1); return out


def sample_starts(data, ptr, size, key, batch_size, hist, horizon=None):
    T,N=data['s'].shape[:2]; k_age,k_env=jax.random.split(key)
    age=jax.random.randint(k_age,(batch_size,),0,jnp.maximum(size,1))
    env=jax.random.randint(k_env,(batch_size,),0,N)
    row=(ptr-size+age)%T; ep=data['ep_step'][row,env]
    back=jnp.minimum(jnp.arange(hist-1,-1,-1)[None,:],ep[:,None]); back=jnp.minimum(back,age[:,None])
    hrows=(ptr-size+age[:,None]-back)%T; henv=jnp.broadcast_to(env[:,None],hrows.shape)
    start={name:data[name][row,env] for name in ('s','a_prev','xi','f')}
    hist_d={name:data[name][hrows,henv] for name in ('s','a_prev','xi','f')}
    return start,hist_d


class JAXReplay:
    def __init__(self,T,N):
        self.T,self.N=T,N; self.ptr=0; self.size=0
        self.data={
            "s":jnp.zeros((T,N,STATE_DIM),jnp.float32),"s1":jnp.zeros((T,N,STATE_DIM),jnp.float32),
            "a":jnp.zeros((T,N,29),jnp.float32),"a_prev":jnp.zeros((T,N,29),jnp.float32),
            "xi":jnp.zeros((T,N,XI_DIM),jnp.float32),"r":jnp.zeros((T,N),jnp.float32),
            "f":jnp.zeros((T,N),jnp.int32),
            "ep_step":jnp.zeros((T,N),jnp.int32),"done":jnp.zeros((T,N),jnp.bool_),
        }

    def add(self, transition):
        self.add_many([transition])

    def add_many(self, transitions):
        """Write a collection batch with one full-buffer functional update."""
        if not transitions:
            return
        batch={name:jnp.stack([tr[name] for tr in transitions]) for name in self.data}
        self.add_batch(batch)

    def add_batch(self, batch):
        """Write an already-stacked scan output without rebuilding its arrays."""
        n=batch['s'].shape[0]
        if n==0:
            return
        rows=(self.ptr+jnp.arange(n,dtype=jnp.int32))%self.T
        self.data={name:value.at[rows].set(batch[name]) for name,value in self.data.items()}
        self.ptr=(self.ptr+n)%self.T; self.size=min(self.size+n,self.T)

    def _row(self,age): return (self.ptr-self.size+age)%self.T

    def sample_windows(self,key,B,k):
        k0,k1=jax.random.split(key); max_t=max(self.size-k+1,1); t0=jax.random.randint(k0,(B,),0,max_t); env=jax.random.randint(k1,(B,),0,self.N)
        ages=t0[:,None]+jnp.arange(k); rows=self._row(ages); e=jnp.broadcast_to(env[:,None],rows.shape)
        out={key:self.data[key][rows,e] for key in ("s","s1","a","xi","done")}; d=out["done"].astype(jnp.float32); out["mask"]=jnp.cumprod(jnp.concatenate((jnp.ones((B,1)),1-d[:,:-1]),-1),-1); return out

    def sample_starts(self,key,B,hist,horizon=None):
        if self.size==0: raise ValueError("cannot sample empty replay")
        k_age,k_env=jax.random.split(key)
        age=jax.random.randint(k_age,(B,),0,self.size); env=jax.random.randint(k_env,(B,),0,self.N)
        row=self._row(age); ep=self.data["ep_step"][row,env]; back=jnp.minimum(jnp.arange(hist-1,-1,-1)[None,:],ep[:,None]); back=jnp.minimum(back,age[:,None]); hrows=self._row(age[:,None]-back); henv=jnp.broadcast_to(env[:,None],hrows.shape)
        start={key:self.data[key][row,env] for key in ("s","a_prev","xi","f")}; hist_d={key:self.data[key][hrows,henv] for key in ("s","a_prev","xi","f")}; return start,hist_d
