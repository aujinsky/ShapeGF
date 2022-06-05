import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, cfg, cfgmodel):
        super().__init__()
        self.cfg = cfg
        self.cfgmodel = cfgmodel

        self.inp_dim = cfgmodel.inp_dim
        self.use_bn = getattr(cfgmodel, "use_bn", False)
        self.use_ln = getattr(cfgmodel, "use_ln", False)
        self.dims = cfgmodel.dims

        curr_dim = self.inp_dim
        layers = []
        for hid in self.dims:
            linear = nn.Linear(curr_dim, hid)
            if self.use_bn:
                bn = nn.BatchNorm1d(hid)
            else:
                bn = nn.Identity()
            if self.use_ln:
                ln = nn.LayerNorm(hid)
            else:
                ln = nn.Identity()
            curr_dim = hid
            layers.append(
                nn.Sequential(
                    linear,
                    bn,
                    ln
                )
            )
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(curr_dim, 1)

    def forward(self, z=None, bs=None, return_all=False):
        if z is None:
            assert bs is not None
            z = torch.randn(bs, self.inp_dim).cuda()
        y = z
        for layer in self.layers:
            y = F.leaky_relu(layer(y), 0.2)
        y = self.out(y)

        if return_all:
            return {
                'x': y
            }
        else:
            return y
