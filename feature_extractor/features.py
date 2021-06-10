import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats

from feature_extractor.utils import set_domain, compute_timestamps_array

@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_mean(signal):
    """
    expects a 1d numpy array
    returns the mean of the array
    """
    return np.mean(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_median(signal):
    """
    expects a 1d numpy array
    returns the median of the array
    """
    return np.median(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_max(signal):
    """
    expects a 1d numpy array
    returns the max value of the array
    """
    return np.max(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_min(signal):
    """
    expects a 1d numpy array
    returns the min value of the array
    """
    return np.min(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_std(signal):
    """
    expects a 1d numpy array
    returns the standard deviation of the array
    """
    return np.std(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_variance(signal):
    """
    expects a 1d numpy array
    returns the variance of the array
    """
    return np.var(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_range(signal):
    """
    expects a 1d numpy array
    returns range between the max and min of the array
    """
    return get_max(signal) - get_min(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_iqr(signal):
    """
    expects a 1d numpy array
    returns the intra quartile range between 25 and 75 (default values) of the array.
    """
    return sp.stats.iqr(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_rms(signal):
    """
    expects a 1d numpy array
    returns the root mean squared "value" of the array.
    """
    return np.sqrt(np.sum(signal ** 2) / len(signal))


@set_domain(domain=["statistical", "time"], input=["1d array"])
def get_energy(signal):
    """
    expects a 1d numpy array
    returns the "energy" of the array.
    """
    return np.sum(signal ** 2) / len(signal)


@set_domain(domain=["statistical", "time"], input=["1d array"])
def zero_crossings(signal):
    """
    expects a 1d numpy array
    returns the zero crossings of the array.
    """
    return float(len(np.where(np.diff(np.sign(signal)))))


#signal magnitude vector
@set_domain(domain=["statistical", "time"], input=["2d array"])
def get_smv(signal: np.array):
    """
    expects a 2d numpy array with the format: [samples, components]
    """
    x, y, z = np.split(signal, 3, axis=1)
    return np.sqrt(np.sum(x ** 2) + np.sum(y ** 2) + np.sum(z ** 2))


# TODO: Flatten
@set_domain(domain=["time"], input=["1d array"])
def get_autocorr(signal):
    """
    expects a 1d numpy array 
    computes the autocorrelation of the signal
    """
    return float(np.correlate(signal, signal))

# TODO: VER
# @set_domain(domain=["time"], input=["1d array", "fs"])
# def get_centroid(signal, fs):
#     """
#     expects a 1d numpy array
#     computes the centroid along the time axis
#     """
#     pass


@set_domain(domain=["time"], input=["1d array"])
def get_negative_turning_count(signal):
    """
    expects a 1d numpy array
    """
    signal_diff = np.diff(signal)
    signal_diff_idxs = np.arange(len(signal_diff[:-1]))

    negative_turning_pts = np.where((signal_diff[signal_diff_idxs + 1] > 0) & \
                                    (signal_diff[signal_diff_idxs] < 0))[0]

    return len(negative_turning_pts)


@set_domain(domain=["time"], input=["1d array"])
def get_positive_turning_count(signal):
    """
    expects a 1d numpy array
    """
    signal_diff = np.diff(signal)
    signal_diff_idxs = np.arange(len(signal_diff[:-1]))

    positive_turning_pts = np.where((signal_diff[signal_diff_idxs + 1] < 0) & \
                                    (signal_diff[signal_diff_idxs] > 0))[0]

    return len(positive_turning_pts)


@set_domain(domain=["time"], input=["1d array"])
def get_mean_diff(signal):
    """
    expects a 1d numpy array
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
    signal_diff = np.sign(signal) 
    return np.sum(np.abs(np.diff(signal)))


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