# from inspect import getmembers, isfunction
#
# from feature_extractor.features import features
# import numpy as np

# from matplotlib import pyplot as plt

# def test_outputs_shape():
#    fs = 1600
#    fake_data = np.random.random((1600, 3))
#    # plt.plot(fake_data)
#    # plt.show()
#    x, y, z = np.split(fake_data, 3, axis=1)
#
#    functions = {name: func for name, func in getmembers(features, isfunction) if name.startswith("get", 0, 3)}
#
#    for f_name, f in functions.items():
#        input_attr = getattr(f, "input")
#
#        if "1d array" in input_attr:
#            for comp, comp_name in zip([x, y, z], ["x", "y", "z"]):
#                # out.append(f(comp.flatten(), fs=fs))
#                 out = f(comp.flatten(), fs=fs)
#                 print(f_name, comp_name, out)
#
#        elif "2d array" in input_attr:
#            # out.append(f(fake_data, fs=fs))
#            out = f(fake_data, fs=fs)
#            print(f_name, out)
#
# # test_outputs_shape()

