import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return


class NoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LobiReFT(h) = h + R^T(b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class Rank1Intervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Rank1(h) = h + (h · v + b) w, with v, w in R^d and b a scalar.
    w is zero-initialized, so the edit is 0 at the start of training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        v = torch.empty(self.embed_dim, dtype=dtype)
        torch.nn.init.normal_(v, mean=0.0, std=self.embed_dim ** -0.5)
        self.v = torch.nn.Parameter(v, requires_grad=True)
        self.w = torch.nn.Parameter(
            torch.zeros(self.embed_dim, dtype=dtype), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros((), dtype=dtype), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.v.dtype)
        scale = torch.matmul(cast_base, self.v) + self.b
        delta = scale.unsqueeze(-1) * self.w
        return base + self.dropout(delta).to(base.dtype)


class OneVecIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    OneVec(h) = h + (h · w) w, with a single w in R^d.
    Hidden states aligned with w are pushed farther along w.
    w is drawn from N(0, 1/d). Zero initialization has no gradient.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        w = torch.empty(self.embed_dim, dtype=dtype)
        torch.nn.init.normal_(w, mean=0.0, std=self.embed_dim ** -0.5)
        self.w = torch.nn.Parameter(w, requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.w.dtype)
        scale = torch.matmul(cast_base, self.w)
        delta = scale.unsqueeze(-1) * self.w
        return base + self.dropout(delta).to(base.dtype)


class AddVecIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    AddVec(h) = h + w, with a single w in R^d.
    The same vector is added at every intervened position.
    w is zero-initialized, so the edit is 0 at the start of training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        self.w = torch.nn.Parameter(
            torch.zeros(self.embed_dim, dtype=dtype), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        delta = self.w.to(base.dtype)
        return base + self.dropout(delta)


class SparseAddVecIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    SparseAddVec(h) = h + alpha * w.

    w starts at 0, so the edit starts at 0. A random fraction of its
    coordinates (default 25%) is trainable. The rest stay 0. alpha starts
    at 1. It must not start at 0, because then both gradients vanish.
    """
    def __init__(self, **kwargs):
        fraction = float(kwargs.pop("trainable_fraction", 0.25))
        mask_seed = int(kwargs.pop("mask_seed", 42))
        super().__init__(**kwargs, keep_last_dim=True)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        embed_dim = int(self.embed_dim)
        n_trainable = min(embed_dim, max(1, int(round(fraction * embed_dim))))
        generator = torch.Generator()
        generator.manual_seed(mask_seed)
        chosen = torch.randperm(embed_dim, generator=generator)[:n_trainable]
        mask = torch.zeros(embed_dim, dtype=torch.bool)
        mask[chosen] = True
        self.register_buffer("w_frozen", torch.zeros(embed_dim, dtype=dtype))
        self.register_buffer("mask", mask)
        self.w_trainable = torch.nn.Parameter(torch.zeros(n_trainable, dtype=dtype), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones((), dtype=dtype), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def steering_vector(self):
        values = self.w_frozen.to(self.w_trainable.dtype)
        index = self.mask.nonzero(as_tuple=False).squeeze(-1)
        return values.index_copy(0, index, self.w_trainable.to(values.dtype))

    def forward(
        self, base, source=None, subspaces=None
    ):
        delta = self.alpha * self.steering_vector()
        return base + self.dropout(delta).to(base.dtype)


class OrthoAddVecIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    OrthoAddVec_i(h) = h + s_i r_i.

    r_1, ..., r_n are orthonormal columns, kept that way by an orthogonal
    parametrization. s_i are free scales, so the trained vectors w_i = s_i r_i
    stay orthogonal to each other and can have different lengths.
    Scales start at 0, so every edit starts at 0.
    Set `active` to choose which vector is added.
    """
    def __init__(self, **kwargs):
        n_vectors = kwargs.pop("n_vectors", 32)
        super().__init__(**kwargs, keep_last_dim=True)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        rotate_layer = LowRankRotateLayer(self.embed_dim, n_vectors, init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.scales = torch.nn.Parameter(torch.zeros(n_vectors, dtype=dtype), requires_grad=True)
        self.active = 0
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def vectors(self):
        # (n_vectors, d). Columns of rotate_layer.weight are orthonormal.
        # Keep this product in float32. The orthogonal map is not stable in float16.
        directions = self.rotate_layer.weight.T
        scales = self.scales.to(directions.dtype)
        return scales[:, None] * directions

    def forward(
        self, base, source=None, subspaces=None
    ):
        w = self.vectors()[self.active]
        return base + self.dropout(w).to(base.dtype)


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))

