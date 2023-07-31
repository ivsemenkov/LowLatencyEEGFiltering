from __future__ import annotations
import random
from cmath import exp
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import numpy.typing as npt


def complex_randn() -> complex:
    """Generate random complex number with Re and Im sampled from N(0, 1)"""
    return random.gauss(0, 1) + 1j * random.gauss(0, 1)


class SignalGenerator(Protocol):
    def step(self) -> complex:
        """Generate single noise sample"""
        ...


@dataclass
class MatsudaParams:
    A: float
    freq: float
    sr: float

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)


@dataclass
class SingleRhythmModel:
    mp: MatsudaParams
    sigma: float
    x: complex = 0

    def step(self) -> complex:
        """Update model state and generate measurement"""
        self.x = self.mp.Phi * self.x + complex_randn() * self.sigma
        return self.x


def collect(signal_generator: SignalGenerator, n_samp: int) -> npt.NDArray[np.number]:
    data = []
    for _ in range(n_samp):
        data_point = signal_generator.step()
        assert data_point is not None
        data.append(data_point)
    return np.array(data)
