import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, cfgmodel):
        super(Encoder, self).__init__()
        self.zdim = cfgmodel.zdim
        self.out = nn.Parameter(torch(randn(1, self.zdim), requires_grad=True))
    def forward(self, x):
        dim0 = x.shape[0]
        return self.out.expand(dim0, -1), self.out.expand(dim0, -1)
    
if __name__ == "__main__":
    class Cfg():
        zdim = 512
    enc = Encoder(Cfg)
    m, v = enc.forward(torch.zeros(256))
    print(enc.zdim, enc.out.shape)
    print(m.shape, v.shape)
