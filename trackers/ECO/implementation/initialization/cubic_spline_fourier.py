import numpy as np

def cubic_spline_fourier(f, a):
    f[np.where(f==0)[0]] = f[0] - 1000
    bf = -(- 12*a + 12*np.exp(-np.pi*f*2j) + 12*np.exp(np.pi*f*2j) + 6*a*np.exp(-np.pi*f*4j) +
        6*a*np.exp(np.pi*f*4j) + f*(np.pi*np.exp(-np.pi*f*2j)*12j) - f*(np.pi*np.exp(np.pi*f*2j)*12j) +
        a*f*(np.pi*np.exp(-np.pi*f*2j)*16j) - a*f*(np.pi*np.exp(np.pi*f*2j)*16j) +
        a*f*(np.pi*np.exp(-np.pi*f*4j)*4j) - a*f*(np.pi*np.exp(np.pi*f*4j)*4j) - 24)/(16*f**4*np.pi**4)
    bf[np.where(f == f[0]-1000)[0]] = 1
    return bf