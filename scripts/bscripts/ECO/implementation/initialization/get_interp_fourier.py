import numpy as np
from .cubic_spline_fourier import cubic_spline_fourier
from ..runfiles import settings

def get_interp_fourier(sz):
    params = settings.params
    if (params["interpolation_method"] == 'bicubic'):
        a = params["interpolation_bicubic_a"]
        interp1_fs = np.real(1/sz[0] * cubic_spline_fourier(np.array(range(int(-(sz[0]-1)/2), int((sz[0]-1)/2 + 1)), dtype=np.float32)/sz[0], a))
        interp2_fs = np.real(1/sz[1] * cubic_spline_fourier(np.array(range(int(-(sz[1]-1)/2), int((sz[1]-1)/2 + 1)), dtype=np.float32)/sz[1], a))
    else:
            print(params["interpolation_method"])
            raise ValueError("Unknown intorpolation method")

    if (params["interpolation_centering"]):
        interp1_fs = interp1_fs * np.exp(-1j*np.pi / sz[0] * np.array(range(int(-(sz[0]-1)/2), int((sz[0]-1)/2 + 1)), dtype=np.float32))
        interp2_fs = interp2_fs * np.exp(-1j*np.pi / sz[1] * np.array(range(int(-(sz[1]-1)/2), int((sz[1]-1)/2 + 1)), dtype=np.float32))
    if (params["interpolation_windowing"]):
        print("Ops, interpolation_windowing")
    return (interp1_fs, interp2_fs)