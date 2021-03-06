import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats

from feature_extractor.utils.feature_utils import compute_fft
from feature_extractor.utils.feature_utils import compute_time
from feature_extractor.utils.feature_utils import set_domain

# https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
# https://github.com/fraunhoferportugal/tsfel

# Statistical features


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_mean(signal, **kwargs):
    """
    Mean of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.mean(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_median(signal, **kwargs):
    """
    Median of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.median(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_max(signal, **kwargs):
    """
    Max of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.max(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_min(signal, **kwargs):
    """
    Min of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.min(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_std(signal, **kwargs):
    """
    Standard deviation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.std(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_variance(signal, **kwargs):
    """
    Variance of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.var(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_range(signal, **kwargs):
    """
    Range between max and min of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.max(signal) - np.min(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_iqr(signal, **kwargs):
    """
    Intra quartile range of a 1-dimensional array
    Expects a 1d numpy array
    """

    return sp.stats.iqr(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_kurtosis(signal, **kwargs):
    """
    Kurtosis of a 1-dimensional array
    Expects a 1d numpy array
    """

    return sp.stats.kurtosis(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_skewness(signal, **kwargs):
    """
    Kurtosis of a 1-dimensional array
    Expects a 1d numpy array
    """

    return sp.stats.skew(signal)


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_rms(signal, **kwargs):
    """
    Root mean square of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.sqrt(np.sum(signal ** 2) / len(signal))


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_mean_abs_deviation(signal, **kwargs):
    """
    Mean absolute deviation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.mean(np.abs(signal - np.mean(signal)))


@set_domain(domain=["statistical"], input=["1d array"],
            sensors=["accelerometer"])
def get_median_abs_deviation(signal, **kwargs):
    """
    Median absolute deviation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return scipy.stats.median_absolute_deviation(signal, scale=1)

# Time domain features


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_autocorr(signal, **kwargs):
    """
    Autocorrelation of a 1-dimensional array
    Expects a 1d numpy array
    """

    return float(np.correlate(signal, signal))


"""
@set_domain(domain=["time"], input=["2d array"], sensors=["accelerometer"])
def get_trosscorr(signal):
  pass
"""


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_zero_crossings(signal, **kwargs):
    """
    Zero-crossings of a 1-dimensional array
    Expects a 1d numpy array
    """
    # Does not work with Z without subtracting 1
    # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python

    return len(np.where(np.diff(np.signbit(signal)))[0])


@set_domain(domain=["time"], input=["2d array"],
            sensors=["accelerometer"])
def get_svm(signal, **kwargs):
    """
    Signal vector magnitude of a 2-dimensional array
    Expects a 2d numpy array with the format: [samples, components]
    """

    return np.sum(np.sqrt(np.sum(signal**2, axis=1)))/signal.shape[-1]


@set_domain(domain=["time"], input=["2d array"],
            sensors=["accelerometer"])
def get_sma(signal, **kwargs):
    """
    Signal magnitude area of a 2-dimensional array
    Expects a 2d array

    Reference: https://downloads.hindawi.com/journals/mpe/2015/790412.pdf
    """

    return np.sum(np.sum(np.abs(signal), axis=1))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_negative_turning_count(signal, **kwargs):
    """
    Number of negative turnings of a 1-dimensional array
    Expects a 1d numpy array
    """

    signal_diff = np.diff(signal)
    signal_diff_idxs = np.arange(len(signal_diff[:-1]))

    negative_turning_pts = np.where((signal_diff[signal_diff_idxs + 1] > 0) &
                                    (signal_diff[signal_diff_idxs] < 0))[0]

    return len(negative_turning_pts)


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_positive_turning_count(signal, **kwargs):
    """
    Number of positive turnings of a 1-dimensional array
    Expects a 1d numpy array
    """

    signal_diff = np.diff(signal)
    signal_diff_idxs = np.arange(len(signal_diff[:-1]))

    positive_turning_pts = np.where((signal_diff[signal_diff_idxs + 1] < 0) &
                                    (signal_diff[signal_diff_idxs] > 0))[0]

    return len(positive_turning_pts)


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_energy(signal, **kwargs):
    """
    https://dsp.stackexchange.com/questions/3377/calculating-the-total-energy-of-a-signal/3378
    expects a 1d numpy array
    """
    return np.dot(signal, signal) / len(signal)


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_abs_energy(signal, **kwargs):
    """
    Energy of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.dot(signal, signal)

# TODO: VER
# @set_domain(domain=["time"], input=["1d array", "fs"])
# def get_centroid(signal, fs):
#     """
#     expects a 1d numpy array
#     computes the centroid along the time axis
#     """
#     pass


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_mean_diff(signal, **kwargs):
    """
    Mean of the differences between values of a 1-dimensional array
    Expects a 1d numpy array
    """

    return np.mean(np.diff(signal))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_median_abs_diff(signal, **kwargs):
    """
    expects a 1d numpy array
    """

    return np.median(np.abs(np.diff(signal)))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_median_diff(signal, **kwargs):
    """
    expects a 1d numpy array
    """

    return np.median(np.diff(signal))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_distance(signal, **kwargs):
    """
    expects a 1d numpy array
    """

    signal_diff = np.sign(signal)
    return np.sum([np.sqrt(1 + signal_diff ** 2)])


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_sum_abs_diff(signal, **kwargs):
    """
    expects a 1d numpy array
    """

    signal_sign = np.sign(signal)
    return np.sum(np.abs(np.diff(signal_sign)))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_slope(signal, **kwargs):
    """
    Expects a 1D numpy array
    """

    t = np.linspace(0, len(signal) - 1, len(signal))
    return np.polyfit(t, signal, 1)[0]


@set_domain(domain=["time"], input=["1d array", "fs"],
            sensors=["accelerometer"])
def get_auc(signal, **kwargs):
    """
    Expects a 1D numpy array
    """
    fs = kwargs.get("fs")
    t = compute_time(len(signal), fs)

    return np.sum(.5 * np.diff(t) * np.abs(signal[:-1] + signal[1:]))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_abs_sum_of_changes(signal, **kwargs):
    """
    Sum over the abstolute value of consecutive changes in the signal
    """

    return np.sum(np.abs(np.diff(signal)))


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_count_above_mean(signal, **kwargs):
    """
    Number of values in x that are higher then the mean of x
    """
    m = np.mean(signal)
    return len(np.where(signal > m)[0])


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_count_below_mean(signal, **kwargs):
    """
    Number of values in x that are lower then the mean of x
    """
    m = np.mean(signal)
    return len(np.where(signal < m)[0])


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_first_location_of_maximum(signal, **kwargs):
    """
    First location of the max value of signal (relative to the length of x)
    """
    return np.argmax(signal) / len(signal)


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_first_location_of_minimum(signal, **kwargs):
    """
    First location of the min value of signal (relative to the length of x)
    """

    return np.argmin(signal) / len(signal)


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_last_location_of_maximum(signal, **kwargs):
    """
    First location of the max value of signal (relative to the length of x)
    """
    return 1.0 - np.argmax(signal[::-1]) / len(signal)


@set_domain(domain=["time"], input=["1d array"],
            sensors=["accelerometer"])
def get_last_location_of_minimum(signal, **kwargs):
    """
    First location of the min value of signal (relative to the length of x)
    """

    return 1.0 - np.argmin(signal[::-1]) / len(signal)


# Frequency domain features (Spectral domain?)
# "All frequency domain features require preprocessing and FFT"


@set_domain(domain=["frequency"], input=["1d array", "fs"],
            sensors=["accelerometer", "audio"])
def get_spectral_energy(signal, **kwargs):
    """
    The energy of the signal can be computed as the squared sum of its spectral
    coefficients normalized by the length of the sample window;
    """
    fs = kwargs.get("fs")
    _, mag_freqs = compute_fft(signal, fs)
    return np.dot(mag_freqs, mag_freqs) / len(signal)


# Information/Spectral entropy


@set_domain(domain=["frequency"], input=["1d array", "fs"],
            sensors=["accelerometer", "audio"])
def get_spectral_distance(signal, **kwargs):
    fs = kwargs.get("fs")
    _, mag_freqs = compute_fft(signal, fs)
    cumsum_mag_freqs = np.cumsum(mag_freqs)

    # Computing the linear regression?
    points_y = np.linspace(0, cumsum_mag_freqs[-1], len(cumsum_mag_freqs))

    return np.sum(points_y - cumsum_mag_freqs)


# @set_domain(domain=["frequency"], input=["1d array", "fs"],
#             sensors=["accelerometer", "audio"])
# def get_fundamental_frequency(signal, **kwargs):
#     fs = kwargs.get("fs")
#     signal = signal - np.mean(signal)
#     freqs, freqs_mag = compute_fft(signal, fs)
#
#     peaks = scipy.signal.find_peaks(freqs_mag, height=max(freqs)*.3)[0]
#
#     peaks = peaks[peaks != 0]
#     if not list(peaks):
#         f0 = 0
#     else:
#         f0 = peaks[min(peaks)]
#
#     return f0
