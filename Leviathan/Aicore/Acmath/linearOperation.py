import jax
import jax.numpy as jnp 

@jax.jit
def dot (w : jax.Array, b : jax.Array) : 
    return jnp.matmul(w,b)

@jax.jit
def multiple (x : jax.Array , y : jax.Array) :
    return x * y 

@jax.jit
def add (x : jax.Array , y : jax.Array) :
    return x + y 

@jax.jit
def substract (x : jax.Array, y : jax.Array) :
    return x - y 

@jax.jit 
def divide (x : jax.Array,y : jax.Array) :
    return x / y

