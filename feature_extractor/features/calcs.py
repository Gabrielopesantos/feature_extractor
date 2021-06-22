import os
from inspect import getmembers, isfunction
from multiprocessing import Pool
from typing import Union, Tuple, List, Dict, Callable

import numpy as np

from feature_extractor.features import features


class Extractor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # Check if any of the needed variables was not passed

    def __call__(self, window_init_end):
        start_idx, end_idx = window_init_end

        window_features = np.empty((self.n_features_out))

        x, y, z, labels = np.split(self.data[start_idx: end_idx], 4, axis=1)

        n: int = 0
        for i_f, (f_name, f) in enumerate(self.features_dict.items()):
            inpt_attr = getattr(f, "input", [])

            if not inpt_attr:
                continue
            elif "1d array" in inpt_attr:
                for comp in [x, y, z]:
                    window_features[n] = f(comp.flatten(), fs=self.fs)
                    n += 1

            elif "2d array" in inpt_attr:
                window_features[n] = f(np.hstack([x, y, z]), fs=1600)
                n += 1

        window_features[-1] = 1 if np.sum(labels) == len(labels) else 0
        return np.reshape(window_features, (1, -1))


def get_features(domain: Tuple[str, ...] = ("time", "statistical", "frequency")) -> Dict[str, Callable]:
    global features

    functions = getmembers(features, isfunction)
    return {f_name: f for f_name, f in functions if f_name.startswith("get", 0, 3) \
            and any(True if d in domain else False for d in getattr(f, "domain"))}


def get_number_out_features(input_features: int, domain: Tuple[str, ...]) -> int:
    features = get_features(domain=domain)

    n: int = 0
    for f_name, f in features.items():
        inpt = getattr(f, "input")

        if "1d array" in inpt:
            n += input_features - 1

        elif "2d array" in inpt:
            n += 1

    # Label
    n += 1

    return n


#
# def extractor_function(**kwargs):
#     n_features_out = kwargs.get("n_features_out")
#     fs = kwargs.get("fs")
#     features_dict = kwargs.get("features_dict")
#     data = kwargs.get("data")
#
#     def inner(window_init_end):
#         start_idx, end_idx = window_init_end
#         nonlocal n_features_out
#         nonlocal fs
#         nonlocal features_dict
#         nonlocal data
#
#         window_features = np.empty((n_features_out))
#
#         x, y, z, labels = np.split(data[start_idx: end_idx], 4, axis=1)
#
#         n: int = 0
#         for i_f, (f_name, f) in enumerate(features_dict.items()):
#             inpt_attr = getattr(f, "input", [])
#
#             if not inpt_attr:
#                 print("Entrei aqui")
#                 continue
#             elif "1d array" in inpt_attr:
#                 for comp in [x, y, z]:
#                     window_features[n] = f(comp.flatten(), fs=fs)
#                     n += 1
#
#             elif "2d array" in inpt_attr:
#                 window_features[n] = f(np.hstack([x, y, z]), fs=1600)
#                 n += 1
#
#         window_features[-1] = 1 if np.sum(labels) == len(labels) else 0
#         return np.reshape(window_features, (1, -1))
#
#     return inner
#
#
# def _extract(window_init_end, data, n_features_out, fs, features_dict):
#     start_idx, end_idx = window_init_end
#
#     window_features = np.empty((n_features_out))
#
#     x, y, z, labels = np.split(data[start_idx: end_idx], 4, axis=1)
#
#     n: int = 0
#     for i_f, (f_name, f) in enumerate(features_dict.items()):
#         inpt_attr = getattr(f, "input", [])
#
#         if not inpt_attr:
#             print("Entrei aqui")
#             continue
#         elif "1d array" in inpt_attr:
#             for comp in [x, y, z]:
#                 window_features[n] = f(comp.flatten(), fs=fs)
#                 n += 1
#
#         elif "2d array" in inpt_attr:
#             window_features[n] = f(np.hstack([x, y, z]), fs=1600)
#             n += 1
#
#     window_features[-1] = 1 if np.sum(labels) == len(labels) else 0
#     return np.reshape(window_features, (1, -1))


def extract_features(data: np.ndarray, windows_indices: Tuple[Tuple[int, int], ...],
                     fs: int = 100, domain: Tuple[str] = ("time", "statistical", "frequency"),
                     n_jobs=None) -> np.ndarray:
    n_jobs = os.cpu_count() if n_jobs is None else n_jobs
    features_dict = get_features(domain=domain)
    n_features_out = get_number_out_features(input_features=data.shape[-1], domain=domain)

    extracted = np.empty((len(windows_indices), n_features_out))

    pool = Pool(n_jobs)
    extractor = Extractor(fs=fs, data=data, features_dict=features_dict, n_features_out=n_features_out)
    try:
        extracted = np.concatenate(list(pool.map(extractor, windows_indices)), axis=0)
        # extracted = np.concatenate(list(pool.map(extractor_function(data=data, n_features_out=n_features_out, features_dict=features_dict, fs=fs), windows_indices)), axis=0)
        # extracted = np.concatenate(list(pool.map(lambda  x: _extract(x, data, n_features_out, fs, features_dict), windows_indices)), axis=0)
        # extracted = pool.map(lambda x: _extract(x, data, n_features_out, fs, features_dict), windows_indices)
    finally:
        pool.close()
        pool.join()

    return extracted