from dataclasses import dataclass
import jax.numpy as jnp
import jax
from jax import random as jrandom
import equinox as eqx
import equinox.nn as enn
from jax import vmap, jit, grad, lax
from typing import List, Callable
from functools import reduce

@dataclass
class MLPConfig:
    in_dim: int = 2
    out_dim: int = 1
    width: int = 40
    depth: int = 2 # note: depth is the #(linear layers), and #(hidden layers) = #(linear layers) - 1.
    activation: Callable = jax.nn.silu
    
class MLP(eqx.Module):

    layers: List[enn.Linear]
    activation: Callable

    def __init__(self, config, key):
        keys = jrandom.split(key, config.depth)
        shp = [config.in_dim] + [config.width]*(config.depth-1) + [config.out_dim]
        self.activation = config.activation
        self.layers = [enn.Linear(shp[i], shp[i+1], key=keys[i]) for i in range(config.depth)]
        
    def __call__(self, x):
        out = reduce(lambda acc, layer: self.activation(vmap(layer)(acc)), self.layers[:-1], x)
        return vmap(self.layers[-1])(out)
        
    
@dataclass
class RNNConfig:
    input_dim: int = 2
    output_dim: int = 1
    hidden_dim: int = 40
    
    
class RNN(eqx.Module):
    hidden_dim: int
    Wh: enn.Linear
    Wx: enn.Linear
    Wy: enn.Linear
    act: Callable

    def __init__(self, config, key):
        wh_key, wx_key, wy_key = jrandom.split(key, 3)
        self.hidden_dim = config.hidden_dim
        self.Wh = enn.Linear(config.hidden_dim, config.hidden_dim, key=wh_key)
        self.Wx = enn.Linear(config.input_dim, config.hidden_dim, key=wx_key)
        self.Wy = enn.Linear(config.hidden_dim, config.output_dim, key=wy_key)
        self.act = jax.nn.sigmoid
    
    def __call__(self, x):
        # x shape: (batch size, sequence length, input_dim)
        batch_size = x.shape[0]

        hidden = jnp.zeros((batch_size, self.hidden_dim))

        def rnn_step(h, input):
            h = self.act(self.Wh(h) + self.Wx(input))
            return h, self.Wy(h)
        
        def rnn_step_batched(h_batch, input_batch):
            return lax.scan(rnn_step, h_batch, input_batch)

        hidden, out = vmap(rnn_step_batched, in_axes=(0, 0))(hidden, x)
        # out shape: (batch_size, sequence_length, output_dim)
        return out


# let's make a more general RNN where we can have an arbitrarily-complex MLP
# as the hidden state transition function and the output function
@dataclass
class GeneralRNNConfig:
    input_dim: int = 2
    output_dim: int = 1
    hidden_dim: int = 40
    hidden_mlp_depth: int = 2 # this would be 1 hidden layer
    hidden_mlp_width: int = 100
    output_mlp_depth: int = 2 # this would be 1 hidden layer
    output_mlp_width: int = 100
    activation: Callable = jax.nn.silu

class GeneralRNN(eqx.Module):
    hidden_dim: int
    input_dim: int
    hmlp: MLP
    ymlp: MLP

    def __init__(self, config, key):
        key_hmlp, key_ymlp = jrandom.split(key)
        self.hidden_dim = config.hidden_dim
        self.input_dim = config.input_dim
        hmlp_config = MLPConfig(
            config.hidden_dim + config.input_dim, 
            config.hidden_dim, 
            config.hidden_mlp_width, 
            config.hidden_mlp_depth, 
            config.activation
        )
        self.hmlp = MLP(hmlp_config, key_hmlp)
        ymlp_config = MLPConfig(
            config.hidden_dim, 
            config.output_dim, 
            config.output_mlp_width, 
            config.output_mlp_depth, 
            config.activation
        )
        self.ymlp = MLP(ymlp_config, key_ymlp)

    def __call__(self, x, h=None):
        """The transition is given by:
            h_t = f([h_{t-1}, x_t])
            y_t = g(h_t)
        where f and g are MLPs.

        This function takes in the input and the hidden state and 
        returns an output and a hidden state.
        """
        # x shape: (batch_size, input_dim)
        # h shape: (batch_size, hidden_dim)
        if h is None:
            h = jnp.zeros((x.shape[0], self.hidden_dim))
        else:
            assert h.shape[0] == x.shape[0]
            assert h.shape[1] == self.hidden_dim
        assert x.shape[1] == self.input_dim
        hx = jnp.concatenate((h, x), axis=1)
        h = self.hmlp(hx)
        y = self.ymlp(h)
        return y, h
    
    def forward_sequence(self, x):
        """This function takes in a sequence of inputs and returns a sequence of outputs
        as well as the final hidden state."""
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.shape[0]
        hidden = jnp.zeros((batch_size, self.hidden_dim))
        assert x.shape[2] == self.input_dim

        def rnn_step(h, input):
            y, h = self(input, h)
            return h, y

        # out shape: (batch_size, sequence_length, output_dim)
        hidden, out = lax.scan(rnn_step, hidden, jnp.transpose(x, (1, 0, 2)))
        return jnp.transpose(out, (1, 0, 2)), hidden
