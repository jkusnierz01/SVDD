#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RawBoost augmentations (ASVspoof-style). Used offline before W2V/MERT."""

from __future__ import annotations

import copy
import random
from typing import Optional

import numpy as np
from scipy import signal


def _seed_numpy(rng: random.Random) -> None:
    np.random.seed(rng.randint(0, 2**31 - 1))


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def normWav(x, always):
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    b = 1
    for _ in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


def LnL_convolutive_noise(
    x,
    N_f,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    minBiasLinNonLin,
    maxBiasLinNonLin,
    fs,
):
    y = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(0, N_f):
        g_min, g_max = minG, maxG
        if i == 1:
            g_min = minG - minBiasLinNonLin
            g_max = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(
            nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, g_min, g_max, fs
        )
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1)
    )
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


def SSI_additive_noise(
    x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR)
    x = x + noise
    return x


# Default hyperparameters (ASVspoof RawBoost)
RAWBOOST_DEFAULTS = dict(
    N_f=5,
    nBands=5,
    minF=20,
    maxF=8000,
    minBW=100,
    maxBW=1000,
    minCoeff=50,
    maxCoeff=100,
    minG=0,
    maxG=0,
    minBiasLinNonLin=5,
    maxBiasLinNonLin=20,
    P=10,
    g_sd=2,
    SNRmin=10,
    SNRmax=40,
)


def apply_rawboost(
    x: np.ndarray,
    fs: int,
    rng: random.Random,
    alb: Optional[int] = None,
) -> np.ndarray:
    """
    Apply RawBoost augmentation to 1-D float waveform.

    alb modes (ASVspoof convention):
      1 LnL | 2 ISD | 3 SSI | 4 LnL+ISD | 5 LnL+SSI | 6 ISD+SSI | 7 all | 8 random(1-7)
    """
    _seed_numpy(rng)
    if alb is None:
        alb = rng.randint(1, 8)

    if alb == 8:
        alb = rng.randint(1, 7)

    x = np.asarray(x, dtype=np.float64)
    p = RAWBOOST_DEFAULTS

    if alb in (1, 4, 5, 7):
        x = LnL_convolutive_noise(
            x,
            N_f=p["N_f"],
            nBands=p["nBands"],
            minF=p["minF"],
            maxF=p["maxF"],
            minBW=p["minBW"],
            maxBW=p["maxBW"],
            minCoeff=p["minCoeff"],
            maxCoeff=p["maxCoeff"],
            minG=p["minG"],
            maxG=p["maxG"],
            minBiasLinNonLin=p["minBiasLinNonLin"],
            maxBiasLinNonLin=p["maxBiasLinNonLin"],
            fs=fs,
        )
    if alb in (2, 4, 6, 7):
        x = ISD_additive_noise(x, P=p["P"], g_sd=p["g_sd"])
    if alb in (3, 5, 6, 7):
        x = SSI_additive_noise(
            x,
            SNRmin=p["SNRmin"],
            SNRmax=p["SNRmax"],
            nBands=p["nBands"],
            minF=p["minF"],
            maxF=p["maxF"],
            minBW=p["minBW"],
            maxBW=p["maxBW"],
            minCoeff=p["minCoeff"],
            maxCoeff=p["maxCoeff"],
            minG=p["minG"],
            maxG=p["maxG"],
            fs=fs,
        )

    return x.astype(np.float32)
