import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats

from feature_extractor.utils import set_domain, compute_timestamps_array

# Statistical features

@set_domain(domain=["statistical"], input=["1d array"])
def get_mean(signal):
    """
    Mean of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.mean(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_median(signal):
    """
    Median of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.median(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_max(signal):
    """
    Max of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.max(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_min(signal):
    """
    Min of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.min(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_std(signal):
    """
    Standard deviation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.std(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_variance(signal):
    """
    Variance of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.var(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_range(signal):
    """
    Range between max and min of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.max(signal) - np.min(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_iqr(signal):
    """
    Intra quartile range of a 1-dimensional array
    Expects a 1d numpy array
    """

    return sp.stats.iqr(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_kurtosis(signal):
    """
    Kurtosis of a 1-dimensional array
    Expects a 1d numpy array
    """

    return sp.stats.kurtosis(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_skewness(signal):
    """
    Kurtosis of a 1-dimensional array
    Expects a 1d numpy array
    """

    return sp.stats.skew(signal)


@set_domain(domain=["statistical"], input=["1d array"])
def get_rms(signal):
    """
    Root mean square of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.sqrt(np.sum(signal ** 2) / len(signal))


@set_domain(domain=["statistical"], input=["1d array"])
def get_mean_abs_deviation(signal):
    """
    Mean absolute deviation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.mean(np.abs(signal - np.mean(signal)))


@set_domain(domain=["statistical"], input=["1d array"])
def get_median_abs_deviation(signal):
    """
    Median absolute deviation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return scipy.stats.median_absolute_deviation(signal, scale=1)

# Time domain features

@set_domain(domain=["time"], input=["1d array"])
def get_autocorr(signal):
    """
    Autocorrelation of a 1-dimensional array
    Expects a 1d numpy array 
    """

    return float(np.correlate(signal, signal))


"""
@set_domain(domain=["time"], input=["2d array"])
def get_crosscorr(signal):
    pass
"""

@set_domain(domain=["time"], input=["1d array"])
def zero_crossings(signal):
    """
    Zero-crossings of a 1-dimensional array
    Expects a 1d numpy array
    """
    # Does not work with Z without subtracting 1

    return float(len(np.where(np.diff(np.sign(signal)))))


@set_domain(domain=["time"], input=["2d array"])
def get_smv(signal):
    """
    Signal magnitude vector of a 2-dimensional array
    Expects a 2d numpy array with the format: [samples, components]
    """

    x, y, z = np.split(signal, 3, axis=1)
    return np.sqrt(np.sum(x ** 2) + np.sum(y ** 2) + np.sum(z ** 2))


@set_domain(domain=["time"], input=["1d array"])
def get_negative_turning_count(signal):
    """
    Number of negative turnings of a 1-dimensional array
    Expects a 1d numpy array
    """

    signal_diff = np.diff(signal)
    signal_diff_idxs = np.arange(len(signal_diff[:-1]))

    negative_turning_pts = np.where((signal_diff[signal_diff_idxs + 1] > 0) & \
                                    (signal_diff[signal_diff_idxs] < 0))[0]

    return len(negative_turning_pts)


@set_domain(domain=["time"], input=["1d array"])
def get_positive_turning_count(signal):
    """
    Number of positive turnings of a 1-dimensional array
    Expects a 1d numpy array
    """

    signal_diff = np.diff(signal)
    signal_diff_idxs = np.arange(len(signal_diff[:-1]))

    positive_turning_pts = np.where((signal_diff[signal_diff_idxs + 1] < 0) & \
                                    (signal_diff[signal_diff_idxs] > 0))[0]

    return len(positive_turning_pts)


# TODO: VER
@set_domain(domain=["time"], input=["1d array"])
def get_energy(signal):
    """
    Energy of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.sum(signal ** 2) / len(signal)


# TODO: VER
# @set_domain(domain=["time"], input=["1d array", "fs"])
# def get_centroid(signal, fs):
#     """
#     expects a 1d numpy array
#     computes the centroid along the time axis
#     """
#     pass


@set_domain(domain=["time"], input=["1d array"])
def get_mean_diff(signal):
    """
    Mean of the differences between values of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.mean(np.diff(signal))


@set_domain(domain=["time"], input=["1d array"])
def get_median_abs_diff(signal):
    """
    expects a 1d numpy array
    """

    return np.median(np.abs(np.diff(signal)))


@set_domain(domain=["time"], input=["1d array"])
def get_median_diff(signal):
    """
    expects a 1d numpy array
    """
    
    return np.median(np.diff(signal))


@set_domain(domain=["time"], input=["1d array"])
def get_distance(signal):
    """
    expects a 1d numpy array
    """

    signal_diff = np.sign(signal) 
    return np.sum([np.sqrt(1 + signal_diff ** 2)])


@set_domain(domain=["time"], input=["1d array"])
def get_sum_abs_diff(signal):
    """
    expects a 1d numpy array
    """

    signal_sign = np.sign(signal) 
    return np.sum(np.abs(np.diff(signal_sign)))


@set_domain(domain=["time"], input=["1d array", "fs"])
def get_total_energy(signal, fs):
    """
    expects a 1d numpy array
    """

    t = compute_timestamps_array(len(signal), fs)

    return np.sum(np.array(signal) ** 2) / (t[-1] - t[0])

@set_domain(domain=["time"], input=["1d array"])
def get_slope(signal):
    """
    Expects a 1D numpy array
    """

    t = np.linspace(0, len(signal) - 1, len(signal))
    return (np.polyfit(t, signal, 1)[0])


@set_domain(domain=["time"], input=["1d array", "fs"])
def get_auc(signal, fs):
    """
    Expects a 1D numpy array
    """

    t = compute_timestamps_array(len(signal), fs)

    return np.sum(.5 * np.diff(t) * np.abs(signal[:-1] + signal[1:]))

# Frequency domain features (Spectral domain?)
#@set_domain(domain=["frequency"], input="1d array")
