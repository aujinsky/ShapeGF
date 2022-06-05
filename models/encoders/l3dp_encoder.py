import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, cfgmodel):
        super(Encoder, self).__init__()
        self.zdim = cfgmodel.zdim
        self.use_deterministic_encoder = cfgmodel.use_deterministic_encoder
        self.input_dim = cfgmodel.input_dim
        self.dim_list = [self.input_dim, 128, 128, 256, 512]
        self.dim_list2 = [512, 256, 128, self.zdim]
        blocks = []
        blocks2 = []
        for i in range(len(self.dim_list)-2):
            blocks.append(nn.Sequential (
                nn.Conv1d(self.dim_list[i], self.dim_list[i+1], 1),
                nn.BatchNorm1d(self.dim_list[i+1]),
                nn.ReLU()
            ))
        blocks.append(nn.Sequential (
            nn.Conv1d(self.dim_list[-2], self.dim_list[-1], 1),
            nn.BatchNorm1d(self.dim_list[-1])
        ))
        self.blocks = nn.ModuleList(blocks)
        if self.use_deterministic_encoder:
            for i in range(len(self.dim_list2)-2):
                blocks2.append(nn.Sequential (
                    nn.Linear(self.dim_list2[i], self.dim_list2[i+1], 1),
                    nn.BatchNorm1d(self.dim_list2[i+1]),
                    nn.ReLU()
                ))
            blocks2.append(nn.Linear(self.dim_list2[-2], self.dim_list2[-1]))
        else:
            for i in range(len(self.dim_list2)-2):
                blocks2.append(nn.ModuleList([
                    nn.Sequential (
                        nn.Linear(self.dim_list2[i], self.dim_list2[i+1], 1),
                        nn.BatchNorm1d(self.dim_list2[i+1]),
                        nn.ReLU()
                    ),
                    nn.Sequential (
                        nn.Linear(self.dim_list2[i], self.dim_list2[i+1], 1),
                        nn.BatchNorm1d(self.dim_list2[i+1]),
                        nn.ReLU()
                    )
                ]))
            blocks2.append(nn.ModuleList(
                [nn.Linear(self.dim_list2[-2], self.dim_list2[-1]),
                nn.Linear(self.dim_list2[-2], self.dim_list2[-1])]
                ))
        self.blocks2 = nn.ModuleList(blocks2)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)  # or is it traspose(0, 1)?
        if self.use_deterministic_encoder:
            for block2 in self.blocks2:
                x = block2(x)
            m, v = x, 0
        else:
            m = x
            v = x
            for block2 in self.blocks2:
                m = block2[0](m)
                v = block2[1](v)
        # print(m.shape)
        return m, v

