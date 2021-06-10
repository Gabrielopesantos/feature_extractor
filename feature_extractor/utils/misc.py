import numpy as np

def compute_timestamps_array(signal_len, fs):
    return np.arange(0, signal_len)/fs

def set_domain(**kwargs):
    def decorate_func(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func

    return decorate_func
