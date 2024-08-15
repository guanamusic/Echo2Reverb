import torch
from scipy import signal


ALLOWED_SYNTHESIS_FILTERBANK_TYPE = ["none", "mel", "octave", "1/3-octave", "merged_octave"]
ALLOWED_SYNTHESIS_WINDOW = {
    "rect": lambda length: torch.ones(length, dtype=torch.float32).to(torch.float32),
    "cosine": lambda length: torch.from_numpy(signal.windows.cosine(length)).to(torch.float32),
    "triangle": lambda length: torch.from_numpy(signal.windows.triang(length)).to(torch.float32),
    "hann": lambda length: torch.from_numpy(signal.windows.hann(length)).to(torch.float32),
}
ALLOWED_NOISE = {
    "gaussian": [0, 1],
    "conv-velvet-gaussian": [0, 3, 4],
}
DIFFUSE_INSENSITIVE_EPSILON = 0.084
DIFFUSE_INSENSITIVE_FN = lambda x: 1 - torch.nn.ReLU()((1 - DIFFUSE_INSENSITIVE_EPSILON) - x)


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
