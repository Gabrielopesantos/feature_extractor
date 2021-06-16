import numpy as np


# Attrs decorator
def set_domain(**kwargs):
    def decorate_func(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func

    return decorate_func


def compute_time(signal_len, fs):
    return np.arange(0, signal_len)/fs


def compute_fft(signal, fs):
    """
    FFT of a signal

    Returns:
    -------
    freqs: nd-array
        Frequency values
    mag_freqs: nd-array
        Amplitude of the frequencies
    """

    mag_freqs = np.abs(np.fft.fft(signal))
    freqs = np.linspace(0, fs, len(signal))
    half = len(signal) // 2

    return freqs[:half], mag_freqs[:half]

