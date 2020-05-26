import math

import torch


class Operator:
    def __init__(self, sr=44100):
        self.sr = sr

    def sample(self, freq, amplitude=1.0, length=None):
        if (type(freq) is not torch.Tensor
                and type(amplitude) is not torch.Tensor):

            if length is None:
                raise ValueError("Can't use scalar frequency and amplitude " +
                                "without specifying output length")
        elif (type(freq) is torch.Tensor
                and type(amplitude) is torch.Tensor):

            if freq.shape[-1] != amplitude.shape[-1]:
                raise ValueError("Amplitude and frequency tensor sizes do " +
                                 "not match in time dimension")
            if len(freq.shape) > 2 or len(amplitude.shape) > 2:
                raise ValueError("Amplitude and frequency tensors can only " +
                                 "contain batch and time dimensions")

        if type(freq) is not torch.Tensor:
            freq = torch.ones(length) * freq

        phase = freq.cumsum(dim=-1) * math.tau / self.sr
        phase = phase - phase[0]

        y = torch.cos(phase) * amplitude
        return y
