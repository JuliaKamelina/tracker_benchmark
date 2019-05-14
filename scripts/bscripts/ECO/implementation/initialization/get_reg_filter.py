import numpy as np
from ..runfiles import settings

def get_reg_filter(sz, target_sz, reg_window_edge):
    params = settings.params

    if np.all(reg_window_edge == []):
        reg_window_edge = params["reg_window_edge"]

    if (params["use_reg_window"]):
        reg_window_power = params["reg_window_power"]

        reg_scale = 0.5*target_sz

        wrg = np.arange(-(sz[0]-1)/2, (sz[0]-1)/2+1)
        wcg = np.arange(-(sz[1]-1)/2, (sz[1]-1)/2+1)
        wrs, wcs = np.meshgrid(wrg, wcg)

        reg_window = (reg_window_edge - params["reg_window_min"]) * \
                     (np.abs(wrs/reg_scale[0])**reg_window_power + np.abs(wcs/reg_scale[1])**reg_window_power) + \
                     params["reg_window_min"]
        
        reg_window_dft = np.fft.fft(np.fft.fft(reg_window, axis=1), axis=0).astype(np.complex64) / np.prod(sz)
        reg_window_dft[np.abs(reg_window_dft) < params["reg_sparsity_threshold"] * np.max(np.abs(reg_window_dft.flatten()))] = 0

        reg_window_sparse = np.real(np.fft.ifft(np.fft.ifft(reg_window_dft, axis=1), axis=0).astype(np.complex64))
        reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(sz) * np.min(reg_window_sparse.flatten()) + params["reg_window_min"]
        reg_window_dft = np.fft.fftshift(reg_window_dft).astype(np.complex64)

        row_idx = np.logical_not(np.all(reg_window_dft==0, axis=1))
        col_idx = np.logical_not(np.all(reg_window_dft==0, axis=0))
        mask = np.outer(row_idx, col_idx)
        reg_filter = np.real(reg_window_dft[mask]).reshape(np.sum(row_idx), -1)
        
        return(reg_filter.T)