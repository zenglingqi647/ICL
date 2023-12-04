import torch
import torch.nn as nn
import numpy as np
class SelfAttention(nn.Module):
    """
    A customized attention layer, following the notation in the paper https://arxiv.org/pdf/2306.04637.pdf#page7

    """
    def __init__(
            self,
            input_dim: int,
            qk_dim: int=None,
            v_dim: int=None,
            num_heads: int=1,
            is_causal: bool=False,
            activation: str="relu"
    ):
        super().__init__()
        # by default, let qk_dim, v_dim, input_dim all be the same
        if qk_dim is None:
            qk_dim = input_dim
        if v_dim is None:
            v_dim = input_dim
        # assert input_dim == qk_dim == v_dim, "Now Assume all dimensions are equal"
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.is_causal = is_causal

        # initialize the weights
        self.K_nets = nn.ModuleList(
            [nn.Linear(input_dim, qk_dim, bias=False) for _ in range(self.num_heads)]
        )
        self.Q_nets = nn.ModuleList(
            [nn.Linear(input_dim, qk_dim, bias=False) for _ in range(self.num_heads)]
        )
        self.V_nets = nn.ModuleList(
            [nn.Linear(input_dim, v_dim, bias=False) for _ in range(self.num_heads)]
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        if activation == "relu":
            self.mask_value = 0.0
        elif activation == "softmax":
            self.mask_value = -float("inf")
        else:
            raise Exception

    def forward(self, x):
        # expects x.shape == (..., N, self.input_dim)
        assert x.shape[-1] == self.input_dim
        N = x.shape[-2]
        output = x # (..., N, self.input_dim)
        for i in range(self.num_heads):
            K_net, Q_net, V_net = self.K_nets[i], self.Q_nets[i], self.V_nets[i]
            K, Q, V = K_net(x), Q_net(x), V_net(x) # (..., N, self.qk_dim); (..., N, self.v_dim)
            similarity = torch.einsum("...nd, ...md -> ...nm", Q, K) # (..., N, N)
            
            # activation
            if self.is_causal:
                masked_idx = ~torch.tril(torch.ones(similarity.shape, dtype=bool))
                similarity[masked_idx] = self.mask_value
            if self.activation == "relu":
                similarity = self.relu(similarity) / N
            elif self.activation == "softmax":
                similarity = self.softmax(similarity / np.sqrt(self.qk_dim))
            
            head_output = torch.einsum("...nm, ...md -> ...nd", similarity, V) # (..., N, self.v_dim)
            output = output + head_output # under the assumption that self.v_dim == self.input_dim
        return output
    

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int=None
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.inner_net = nn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.outer_net = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        return x + self.outer_net(self.relu(self.inner_net(x)))
    

class Transformer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            num_heads: int=1,
            mlp_hidden_dim: int=None,
            is_causal: bool=False,
            activation: str="relu"
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(SelfAttention(input_dim=input_dim, num_heads=num_heads, is_causal=is_causal, activation=activation))
            layers.append(MLP(input_dim=input_dim, hidden_dim=mlp_hidden_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

        


class SelfAttention_Inefficient(nn.Module):
    """
    A customized attention layer, following the notation in the paper https://arxiv.org/pdf/2306.04637.pdf#page7

    """
    default_init_std = 0.1
    def __init__(
            self,
            input_dim: int,
            qk_dim: int=None,
            v_dim: int=None,
            num_heads: int=1,
            WK_init: torch.Tensor=None,
            WQ_init: torch.Tensor=None,
            WV_init: torch.Tensor=None,
    ):
        super().__init__()
        # by default, let qk_dim, v_dim, input_dim all be the same
        if qk_dim is None:
            qk_dim = input_dim
        if v_dim is None:
            v_dim = input_dim
        assert qk_dim == v_dim == input_dim, "Right now assume qk_dim = v_dim = input_dim"
        self.input_dim = input_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads

        # initialize the weights
        WK_init = WK_init.detach() if WK_init is not None else torch.normal(0, self.default_init_std, (self.num_heads, self.input_dim, self.qk_dim))
        self.WK = nn.Parameter(WK_init, requires_grad=True)
        assert self.WK.shape == (self.num_heads, self.input_dim, self.qk_dim)

        WQ_init = WQ_init.detach() if WQ_init is not None else torch.normal(0, self.default_init_std, (self.num_heads, self.input_dim, self.qk_dim))
        self.WQ = nn.Parameter(WQ_init, requires_grad=True)
        assert self.WQ.shape == (self.num_heads, self.input_dim, self.qk_dim)
        
        WV_init = WV_init.detach() if WV_init is not None else torch.normal(0, self.default_init_std, (self.num_heads, self.input_dim, self.v_dim))
        self.WV = nn.Parameter(WV_init, requires_grad=True)
        assert self.WV.shape == (self.num_heads, self.input_dim, self.v_dim)

        self.relu = nn.ReLU()

    def normalized_ReLU(self, x):
        assert x.shape[-1] == self.input_dim
        N = x.shape[-2]
        # x.shape = (..., N, self.input_dim)
        output = self.relu(x) / N
        return output

    def forward(self, x):
        # expects x.shape == (..., N, self.input_dim)
        assert x.shape[-1] == self.input_dim
        N = x.shape[-2]
        K = torch.einsum("...nd, mdk -> ...mnk", x, self.WK) # (..., self.num_heads, N, self.qk_dim)
        Q = torch.einsum("...nd, mdk -> ...mnk", x, self.WQ) # (..., self.num_heads, N, self.qk_dim)
        V = torch.einsum("...nd, mdk -> ...mnk", x, self.WV) # (..., self.num_heads, N, self.v_dim)
        torch.einsum("...")