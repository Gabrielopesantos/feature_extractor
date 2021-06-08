import contextlib

import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats


def set_domain(**kwargs):
    def decorate_func(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
            return func

    return decorate_func

@contextlib.contextmanager
def 


@set_domain(domain=["time"])
def get_mean(signal):
    return np.mean(signal)


def get_median(signal):
    return np.median(signal)


def get_max(signal):
    return np.max(signal)


def get_min(signal):
    return np.min(signal)


def get_std(signal):
    return np.std(signal)


def get_variance(signal):
    return np.var(signal)


def get_range(signal):
    return get_max(signal) - get_min(signal)


def get_iqr(signal):
    return sp.stats.iqr(signal)


def get_rms(signal):
    return np.sqrt(np.sum(signal ** 2) / len(signal))


def get_energy(signal):
    return np.sqrt(signal ** 2) // len(signal)


def zero_crossings(signal):
    return float(len(np.where(np.diff(np.sign(signal)))))

#signal magnitude vector
def get_smv(signal: np.array):
    x, y, z = np.split(signal, 3, axis=1)
    return np.sqrt(np.sum(x ** 2) + np.sum(y ** 2) + np.sum(z ** 2))